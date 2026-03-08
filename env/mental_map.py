"""
MentalMap module: Commander's internal representation of the grid.
Initialized identically to Grid but updated only via field agent observations.
Maintains uncertainty about unexplored areas.
"""

import numpy as np
import networkx as nx
from typing import Dict, List, Tuple, Optional, Set
from copy import deepcopy

from .grid import Grid, Cell, CellType, HazardType, Victim, Building


class MentalMapCell:
    """
    Commander's belief about a cell. Similar to Cell but tracks uncertainty.
    """
    def __init__(self, x: int, y: int):
        self.x = x
        self.y = y
        self.cell_type: Optional[CellType] = None  # Unknown until observed
        self.building_id: Optional[int] = None
        self.blocked: Optional[bool] = None
        self.hazard: Optional[HazardType] = None
        self.fire_intensity: Optional[float] = None
        self.victims: List[Victim] = []  # Known victims
        self.explored: bool = False
        self.last_updated_step: int = -1  # -1 means never updated
        
    @property
    def is_known(self) -> bool:
        """Whether this cell has been observed."""
        return self.explored
    
    @property
    def passable(self) -> Optional[bool]:
        """Return passability if known, else None."""
        if self.cell_type is None or self.blocked is None:
            return None
        return self.cell_type == CellType.ROAD and not self.blocked

    def to_dict(self) -> dict:
        """Convert to dictionary for debugging/logging."""
        return {
            'position': (self.x, self.y),
            'cell_type': self.cell_type.name if self.cell_type else 'UNKNOWN',
            'blocked': self.blocked,
            'hazard': self.hazard.name if self.hazard else 'UNKNOWN',
            'fire_intensity': self.fire_intensity,
            'num_victims': len(self.victims),
            'explored': self.explored,
            'last_updated_step': self.last_updated_step,
        }


class MentalMap:
    """
    Commander's mental model of the grid. 
    - Initialized with the same structure as Grid (layout, buildings)
    - But hazards, victims, and dynamic state are unknown until reported by field agents
    - Tracks uncertainty and staleness of information
    """
    
    def __init__(self, width: int, height: int, seed: int = 42):
        self.width = width
        self.height = height
        self.rng = np.random.default_rng(seed)
        self.cells: Dict[Tuple[int, int], MentalMapCell] = {}
        self.buildings: Dict[int, Building] = {}  # Known building structures
        self.graph = nx.Graph()  # Believed traversable graph
        self.current_step = 0
        
        # Initialize empty grid with unknown cells
        self._init_empty_grid()
    
    def _init_empty_grid(self):
        """Initialize with all cells as unexplored/unknown."""
        for y in range(self.height):
            for x in range(self.width):
                self.cells[(x, y)] = MentalMapCell(x=x, y=y)
                self.graph.add_node((x, y))
        
        # Initial full adjacency (will be updated as we learn about blockages)
        for y in range(self.height):
            for x in range(self.width):
                for dx, dy in [(1, 0), (0, 1), (-1, 0), (0, -1)]:
                    nx_, ny_ = x + dx, y + dy
                    if 0 <= nx_ < self.width and 0 <= ny_ < self.height:
                        self.graph.add_edge((x, y), (nx_, ny_))
    
    def initialize_from_grid(self, grid: Grid):
        """
        Initialize mental map with known static information from the actual grid:
        - Building layouts (structural information that would be known pre-disaster)
        - Cell types (road vs building from city maps)
        
        Does NOT copy dynamic information like victims, fires, or damage.
        """
        # Copy building structures (assumed to be known from maps)
        self.buildings = deepcopy(grid.buildings)
        
        # Copy cell types and building assignments
        for pos, cell in grid.cells.items():
            mental_cell = self.cells[pos]
            mental_cell.cell_type = cell.cell_type
            mental_cell.building_id = cell.building_id
            
            # If it's a building, remove from traversal graph
            if cell.cell_type == CellType.BUILDING:
                if pos in self.graph:
                    self.graph.remove_node(pos)
            
            # Keep dynamic state unknown (blocked, hazards, victims)
            # These will be updated through agent observations
    
    def update_from_observation(self, position: Tuple[int, int], 
                               observation: List[List[dict]], 
                               radius: int,
                               step: int):
        """
        Update mental map from an agent's local observation.
        
        Args:
            position: Agent's current position
            observation: Grid observation from agent (as returned by get_local_observation)
            radius: Observation radius used
            step: Current simulation step
        """
        self.current_step = step
        x, y = position
        
        for dy_idx, obs_row in enumerate(observation):
            for dx_idx, obs_cell in enumerate(obs_row):
                # Convert observation indices to absolute position
                dx = dx_idx - radius
                dy = dy_idx - radius
                abs_pos = (x + dx, y + dy)
                
                if abs_pos not in self.cells:
                    continue
                
                if obs_cell['type'] == 'OUT_OF_BOUNDS':
                    continue
                
                mental_cell = self.cells[abs_pos]
                
                # Update cell information
                mental_cell.explored = True
                mental_cell.last_updated_step = step
                
                # Update hazard information
                if obs_cell['hazard'] != 'NONE':
                    mental_cell.hazard = HazardType[obs_cell['hazard']]
                    mental_cell.fire_intensity = obs_cell.get('fire_intensity', 0.0)
                else:
                    mental_cell.hazard = HazardType.NONE
                    mental_cell.fire_intensity = 0.0
                
                # Update blockage information for roads
                if obs_cell.get('type') == 'ROAD':
                    was_blocked = mental_cell.blocked
                    mental_cell.blocked = obs_cell.get('blocked', False)
                    
                    # Update graph if blockage state changed
                    if mental_cell.blocked and not was_blocked:
                        # Newly blocked
                        if abs_pos in self.graph:
                            self.graph.remove_node(abs_pos)
                    elif not mental_cell.blocked and was_blocked:
                        # Newly unblocked
                        if abs_pos not in self.graph:
                            self.graph.add_node(abs_pos)
                            # Restore edges to adjacent road cells
                            for ddx, ddy in [(1, 0), (0, 1), (-1, 0), (0, -1)]:
                                nb = (abs_pos[0] + ddx, abs_pos[1] + ddy)
                                if nb in self.graph:
                                    self.graph.add_edge(abs_pos, nb)
                
                # Update building collapse status if observed
                if obs_cell.get('type') == 'BUILDING' and obs_cell.get('collapsed', False):
                    building_id = obs_cell.get('building_id')
                    if building_id is not None and building_id in self.buildings:
                        self.buildings[building_id].collapsed = True
                
                # Note: Victim information is updated separately via update_victim_info
    
    def update_victim_info(self, position: Tuple[int, int], 
                          victim_count: int,
                          step: int):
        """
        Update victim information for a specific cell.
        Called when agents report victim sightings.
        
        Args:
            position: Cell position
            victim_count: Number of victims observed
            step: Current simulation step
        """
        if position not in self.cells:
            return
        
        mental_cell = self.cells[position]
        mental_cell.last_updated_step = step
        
        # Simple heuristic: update victim count
        # In a more sophisticated version, we could track individual victims
        # For now, we just know approximately how many victims are at this location
        # This is a simplified representation
    
    def update_building_collapse(self, building_id: int, collapsed: bool, step: int):
        """
        Update knowledge about building collapse status.
        
        Args:
            building_id: ID of the building
            collapsed: Whether the building is collapsed
            step: Current simulation step
        """
        if building_id in self.buildings:
            self.buildings[building_id].collapsed = collapsed
            self.current_step = step
            
            # If collapsed, mark adjacent cells as potentially blocked
            # (This is conservative - actual spillover may vary)
            if collapsed:
                building = self.buildings[building_id]
                for cell_pos in building.cells:
                    # Mark building cell area as explored since collapse is visible
                    if cell_pos in self.cells:
                        self.cells[cell_pos].last_updated_step = step
    
    def get_uncertainty_map(self) -> np.ndarray:
        """
        Return a 2D array representing information staleness/uncertainty.
        
        Returns:
            Array where values indicate steps since last update
            (-1 for never updated, 0 for just updated)
        """
        uncertainty = np.full((self.height, self.width), -1, dtype=int)
        
        for (x, y), cell in self.cells.items():
            if cell.last_updated_step >= 0:
                uncertainty[y, x] = self.current_step - cell.last_updated_step
            else:
                uncertainty[y, x] = -1  # Never explored
        
        return uncertainty
    
    def get_explored_fraction(self) -> float:
        """Return fraction of cells that have been explored."""
        total = len(self.cells)
        explored = sum(1 for cell in self.cells.values() if cell.explored)
        return explored / total if total > 0 else 0.0
    
    def get_local_belief(self, x: int, y: int, radius: int = 1) -> List[List[dict]]:
        """
        Get commander's belief about a local area (similar to grid observation).
        Includes uncertainty indicators.
        """
        obs = []
        for dy in range(-radius, radius + 1):
            row = []
            for dx in range(-radius, radius + 1):
                pos = (x + dx, y + dy)
                if pos in self.cells:
                    c = self.cells[pos]
                    row.append({
                        'type': c.cell_type.name if c.cell_type else 'UNKNOWN',
                        'blocked': c.blocked,
                        'hazard': c.hazard.name if c.hazard else 'UNKNOWN',
                        'fire_intensity': c.fire_intensity if c.fire_intensity else 0.0,
                        'num_victims': len(c.victims),
                        'explored': c.explored,
                        'staleness': self.current_step - c.last_updated_step if c.last_updated_step >= 0 else -1,
                    })
                else:
                    row.append({'type': 'OUT_OF_BOUNDS'})
            obs.append(row)
        return obs
    
    def shortest_path(self, start: Tuple[int, int], 
                     end: Tuple[int, int]) -> Optional[List[Tuple[int, int]]]:
        """
        Compute shortest path based on commander's current belief.
        May differ from actual shortest path if blockages are unknown.
        """
        if start not in self.graph or end not in self.graph:
            return None
        try:
            return nx.astar_path(
                self.graph, start, end,
                heuristic=lambda a, b: abs(a[0]-b[0]) + abs(a[1]-b[1])
            )
        except nx.NetworkXNoPath:
            return None
    
    def get_zone_summary(self, zone_x: int, zone_y: int, 
                        zone_size: int = 10) -> dict:
        """
        Get zone summary based on commander's current knowledge.
        Includes uncertainty metrics.
        """
        victims_known = 0
        fires_known = 0
        blocked_roads_known = 0
        collapsed_known = 0
        cells_explored = 0
        total_cells = 0
        
        for dy in range(zone_size):
            for dx in range(zone_size):
                pos = (zone_x + dx, zone_y + dy)
                if pos not in self.cells:
                    continue
                
                total_cells += 1
                c = self.cells[pos]
                
                if c.explored:
                    cells_explored += 1
                    victims_known += len(c.victims)
                    
                    if c.hazard == HazardType.FIRE:
                        fires_known += 1
                    
                    if c.blocked:
                        blocked_roads_known += 1
                    
                    if c.building_id is not None:
                        b = self.buildings.get(c.building_id)
                        if b and b.collapsed:
                            collapsed_known += 1
        
        exploration_ratio = cells_explored / total_cells if total_cells > 0 else 0.0
        
        return {
            'zone': (zone_x, zone_y),
            'victims_known': victims_known,
            'fires_known': fires_known,
            'blocked_roads': blocked_roads_known,
            'collapsed_buildings': collapsed_known,
            'exploration': exploration_ratio,
            'cells_explored': cells_explored,
            'total_cells': total_cells,
        }
    
    def get_all_known_victims(self) -> List[Tuple[int, int, int]]:
        """
        Return list of known victim locations.
        Returns: List of (x, y, count) tuples
        """
        known_victims = []
        for pos, cell in self.cells.items():
            if cell.explored and len(cell.victims) > 0:
                known_victims.append((pos[0], pos[1], len(cell.victims)))
        return known_victims
    
    def to_summary_dict(self) -> dict:
        """Convert mental map to summary dictionary for logging/debugging."""
        return {
            'dimensions': (self.width, self.height),
            'current_step': self.current_step,
            'exploration_fraction': self.get_explored_fraction(),
            'known_buildings': len(self.buildings),
            'collapsed_buildings': sum(1 for b in self.buildings.values() if b.collapsed),
            'uncertainty_stats': {
                'mean_staleness': float(np.mean([
                    self.current_step - c.last_updated_step 
                    for c in self.cells.values() 
                    if c.last_updated_step >= 0
                ])) if any(c.last_updated_step >= 0 for c in self.cells.values()) else -1,
            }
        }
