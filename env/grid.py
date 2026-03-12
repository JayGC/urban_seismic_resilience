"""
Grid module: Cell, Building, and Grid classes for the urban disaster simulation.
Supports OSM-style topology with buildings, roads, hazards, victims, and dynamic blocking.
"""

import numpy as np
import networkx as nx
from enum import Enum
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Tuple, Set



class CellType(Enum):
    ROAD = 0
    BUILDING = 1


class HazardType(Enum):
    NONE = 0
    FIRE = 1
    DEBRIS = 2


@dataclass
class Victim:
    victim_id: int
    health: float = 100.0  # 0-100; 0 = dead
    rescued: bool = False
    trapped: bool = True
    position: Tuple[int, int] = (0, 0)

    def tick(self, decay_rate: float = 1.0):
        """Reduce health each step if not rescued."""
        if not self.rescued and self.health > 0:
            self.health = max(0.0, self.health - decay_rate)


@dataclass
class Cell:
    x: int
    y: int
    cell_type: CellType = CellType.ROAD
    building_id: Optional[int] = None
    blocked: bool = False
    hazard: HazardType = HazardType.NONE
    fire_intensity: float = 0.0  # 0-100
    victims: List[Victim] = field(default_factory=list)
    explored: bool = False

    @property
    def passable(self) -> bool:
        return self.cell_type == CellType.ROAD and not self.blocked

    def to_obs_code(self) -> int:
        """Encode cell state as integer for raw messaging."""
        code = self.cell_type.value
        if self.blocked:
            code += 10
        code += self.hazard.value * 100
        code += len(self.victims) * 1000
        return code


@dataclass
class Building:
    building_id: int
    cells: List[Tuple[int, int]]  # Grid coords occupied
    integrity: float = 100.0  # 0-100
    height: int = 1  # floors; affects spillover radius
    collapsed: bool = False
    collapse_threshold: float = 20.0
    num_people_inside: int = 0

    def apply_damage(self, damage: float):
        self.integrity = max(0.0, self.integrity - damage)

    def check_collapse(self, rng: np.random.Generator) -> bool:
        """Probabilistic collapse when integrity < threshold."""
        if self.collapsed:
            return False
        if self.integrity <= 0:
            self.collapsed = True
    
                # This will be handled by the Grid's spillover logic
            return True
        if self.integrity < self.collapse_threshold:
            prob = 1.0 - (self.integrity / self.collapse_threshold)
            if rng.random() < prob:
                self.collapsed = True
                return True
        return False


class Grid:
    """
    2D grid representing the urban area. Supports dynamic blocking,
    adjacency graph updates, and pathfinding.
    """

    def __init__(self, width: int, height: int, seed: int = 42):
        self.width = width
        self.height = height
        self.rng = np.random.default_rng(seed)
        self.cells: Dict[Tuple[int, int], Cell] = {}
        self.buildings: Dict[int, Building] = {}
        self.graph = nx.Graph()
        self.victim_counter = 0
        self._init_empty_grid()

    def _init_empty_grid(self):
        """Create an empty grid of road cells with full adjacency."""
        for y in range(self.height):
            for x in range(self.width):
                self.cells[(x, y)] = Cell(x=x, y=y)
                self.graph.add_node((x, y))
        # 4-connected adjacency
        for y in range(self.height):
            for x in range(self.width):
                for dx, dy in [(1, 0), (0, 1), (-1, 0), (0, -1)]:
                    nx_, ny_ = x + dx, y + dy
                    if 0 <= nx_ < self.width and 0 <= ny_ < self.height:
                        self.graph.add_edge((x, y), (nx_, ny_))

    # ---- Topology generation ----

    def generate_city_layout(self, building_density: float = 0.3,
                              min_building_size: int = 2,
                              max_building_size: int = 5):
        """
        Procedurally generate a city-like layout with buildings and roads.
        Uses a simple block-based approach (approximating OSM-style layout).
        """
        bid = 0
        occupied: Set[Tuple[int, int]] = set()

        # Place buildings in a grid-like pattern with roads between
        road_spacing = max_building_size + 2  # ensure roads between blocks
        attempts = int(self.width * self.height * building_density)

        for _ in range(attempts):
            bw = self.rng.integers(min_building_size, max_building_size + 1)
            bh = self.rng.integers(min_building_size, max_building_size + 1)
            bx = self.rng.integers(1, max(2, self.width - bw - 1))
            by = self.rng.integers(1, max(2, self.height - bh - 1))

            cells_to_place = []
            valid = True
            for dy in range(bh):
                for dx in range(bw):
                    pos = (bx + dx, by + dy)
                    if pos in occupied:
                        valid = False
                        break
                    # Keep a 2-cell road margin
                    # Enforce a 2-cell road margin around the building block
                    for mx in range(-2, bw + 2):
                        for my in range(-2, bh + 2):
                            mxpos, mypos = bx + mx, by + my
                            if (mxpos, mypos) in occupied and (mx, my) not in [(dx, dy) for dy in range(bh) for dx in range(bw)]:
                                valid = False
                                break
                        if not valid:
                            break
                    cells_to_place.append(pos)
                if not valid:
                    break

            if valid and cells_to_place:
                height = self.rng.integers(1, 6)
                building = Building(building_id=bid, cells=cells_to_place, height=height)
                self.buildings[bid] = building
                for pos in cells_to_place:
                    occupied.add(pos)
                    self.cells[pos].cell_type = CellType.BUILDING
                    self.cells[pos].building_id = bid
                    # Remove building cells from traversal graph
                    if pos in self.graph:
                        self.graph.remove_node(pos)
                bid += 1

    def place_victims(self, count: int):
        """Place victims on building edge cells (adjacent to at least one road)."""
        for building in self.buildings.values():
            # Find edge cells: building cells with at least one road neighbour
            edge_cells = []
            for (bx, by) in building.cells:
                for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                    nb = (bx + dx, by + dy)
                    if nb in self.cells and self.cells[nb].cell_type == CellType.ROAD:
                        edge_cells.append((bx, by))
                        break  # one road neighbour is enough
            if not edge_cells:
                # Fallback: use any building cell if no edge found
                edge_cells = list(building.cells)
            num_victims = max(1, count // len(self.buildings)) if self.buildings else 0
            num_to_place = min(num_victims, len(edge_cells))
            chosen_indices = self.rng.choice(len(edge_cells), size=num_to_place, replace=True)
            for idx in chosen_indices:
                pos = edge_cells[idx]
                v = Victim(victim_id=self.victim_counter, position=pos)
                building.num_people_inside += 1
                self.victim_counter += 1
                self.cells[pos].victims.append(v)

    def place_fires(self, count: int, intensity_range: Tuple[float, float] = (30, 80)):
        """Place fires randomly in buildings."""
        building_cells = [(x, y) for (x, y), c in self.cells.items()
                         if c.cell_type == CellType.BUILDING]
        if not building_cells:
            return
        count = min(count, len(building_cells))
        chosen = self.rng.choice(len(building_cells), size=count, replace=False)
        for idx in chosen:
            pos = building_cells[idx]
            self.cells[pos].hazard = HazardType.FIRE
            self.cells[pos].fire_intensity = float(
                self.rng.uniform(*intensity_range))

    # ---- Dynamic updates ----

    def block_cell(self, x: int, y: int):
        """Block a road cell (e.g. from building spillover) and update graph."""
        cell = self.cells.get((x, y))
        if cell and cell.cell_type == CellType.ROAD:
            cell.blocked = True
            cell.hazard = HazardType.DEBRIS
            if (x, y) in self.graph:
                self.graph.remove_node((x, y))

    def unblock_cell(self, x: int, y: int):
        """Unblock a road cell and restore graph edges."""
        cell = self.cells.get((x, y))
        if cell and cell.cell_type == CellType.ROAD:
            cell.blocked = False
            self.graph.add_node((x, y))
            for dx, dy in [(1, 0), (0, 1), (-1, 0), (0, -1)]:
                nb = (x + dx, y + dy)
                if nb in self.graph:
                    self.graph.add_edge((x, y), nb)

    def apply_spillover(self, building: Building, radius_factor: float = 1.0):
        """Block road cells adjacent to a collapsed building based on height."""
        # radius = int(building.height * radius_factor)
        radius = 1
        for cx, cy in building.cells:
            for dx in range(-radius, radius + 1):
                for dy in range(-radius, radius + 1):
                    if dx == 0 and dy == 0:
                        continue
                    nx_, ny_ = cx + dx, cy + dy
                    if (nx_, ny_) in self.cells:
                        cell = self.cells[(nx_, ny_)]
                        if cell.cell_type == CellType.ROAD:
                            # Probability decreases with distance
                            dist = abs(dx) + abs(dy)
                            prob = max(0, 1.0 - dist / (radius + 1))
                            if self.rng.random() < prob:
                                self.block_cell(nx_, ny_)

    # ---- Queries ----

    def get_local_observation(self, x: int, y: int, radius: int = 1) -> List[List[dict]]:
        """Return a (2*radius+1) x (2*radius+1) observation window around (x,y)."""
        obs = []
        for dy in range(-radius, radius + 1):
            row = []
            for dx in range(-radius, radius + 1):
                pos = (x + dx, y + dy)
                if pos in self.cells:
                    c = self.cells[pos]
                    cell_info = {
                        'type': c.cell_type.name,
                        'blocked': c.blocked,
                        'hazard': c.hazard.name,
                        'fire_intensity': c.fire_intensity,
                        'num_victims': len([v for v in c.victims if not v.rescued and v.health > 0]),
                        'in_danger': self.is_cell_in_danger(pos),
                        'explored': c.explored,
                        'building_id': c.building_id,
                    }
                    # Add collapse info if it's a building
                    if c.cell_type == CellType.BUILDING and c.building_id is not None:
                        building = self.buildings.get(c.building_id)
                        if building:
                            cell_info['collapsed'] = building.collapsed
                    row.append(cell_info)
                else:
                    row.append({'type': 'OUT_OF_BOUNDS'})
            obs.append(row)
        return obs

    def shortest_path(self, start: Tuple[int, int],
                      end: Tuple[int, int]) -> Optional[List[Tuple[int, int]]]:
        """A* shortest path on the current traversable graph."""
        if start not in self.graph or end not in self.graph:
            return None
        try:
            return nx.astar_path(self.graph, start, end,
                                 heuristic=lambda a, b: abs(a[0]-b[0]) + abs(a[1]-b[1]))
        except nx.NetworkXNoPath:
            return None

    def is_cell_in_danger(self, pos: Tuple[int, int]) -> bool:
        """Check if a cell is in a hazardous building (any cell of that building has fire or debris)."""
        cell = self.cells.get(pos)
        if not cell or cell.building_id is None:
            # Non-building cell: check its own hazard
            return cell is not None and cell.hazard in (HazardType.FIRE, HazardType.DEBRIS)
        # Building cell: check all cells of the same building
        bid = cell.building_id
        building = self.buildings.get(bid)
        if building and building.collapsed:
            return True
        for bx, by in building.cells if building else []:
            bcell = self.cells.get((bx, by))
            if bcell and bcell.hazard in (HazardType.FIRE, HazardType.DEBRIS):
                return True
        return False

    def get_all_victims(self) -> List[Victim]:
        """Return all victims across the grid."""
        victims = []
        for cell in self.cells.values():
            victims.extend(cell.victims)
        return victims

    def get_zone_summary(self, zone_x: int, zone_y: int,
                         zone_size: int = 10) -> dict:
        """Coarse zone summary for commander observation."""
        fires = 0
        blocked_roads = 0
        collapsed = 0
        victims = 0

        for dy in range(zone_size):
            for dx in range(zone_size):
                pos = (zone_x + dx, zone_y + dy)
                if pos not in self.cells:
                    continue
                c = self.cells[pos]
                if c.hazard == HazardType.FIRE:
                    fires += 1
                if c.blocked:
                    blocked_roads += 1
                if c.building_id is not None:
                    b = self.buildings.get(c.building_id)
                    if b and b.collapsed:
                        collapsed += 1
                        if b.num_people_inside > 0:
                            victims += b.num_people_inside

        return {
            'zone': (zone_x, zone_y),
            'fires': fires,
            'victims': victims,
            'blocked_roads': blocked_roads,
            'collapsed_buildings': collapsed,
        }
