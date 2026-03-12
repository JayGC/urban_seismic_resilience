"""
Commander agents: coordinate field agents by issuing task assignments.
- HeuristicCommander: rule-based prioritization
- LLMCommander: uses GPT/Claude API for strategic planning
"""

import json
import re
from typing import Dict, List, Tuple, Optional, Any
from .messages import Message, MessageBus, MessageType, make_task_assignment
from env.grid import Victim

try:
    from env.mental_map import MentalMap
    from env.grid import HazardType, CellType
except ImportError:
    from ..env.mental_map import MentalMap
    from ..env.grid import HazardType, CellType

class CommanderAgent:
    """Base commander interface."""

    def __init__(self, agent_ids: List[str]):
        self.agent_ids = agent_ids
        self.agent_reports: Dict[str, dict] = {}  # agent_id -> last known info
        self.mental_map: Optional[MentalMap] = None  # Commander's belief about the grid
        self.zone_data: List[dict] = []
        self.step = 0
        self.assignments: Dict[str, dict] = {}  # agent_id -> current assignment
        self.replan_log: List[dict] = []
        self.field_agents: Dict[str, Any] = {}  # ref to actual agent objects for direct status checks
    
    def initialize_mental_map(self, grid_width: int, grid_height: int, grid, seed: int = 42):
        """Initialize mental map with same structure as actual grid."""
        self.mental_map = MentalMap(grid_width, grid_height, seed)
        # Initialize with static structural information (buildings, roads)
        self.mental_map.initialize_from_grid(grid)

    def update_mental_map(self, messages: List[Message], env=None):
        """Update internal model from agent reports with exact coordinates."""
        for msg in messages:
            if msg.msg_type in (MessageType.REPORT, MessageType.EMERGENCY):
                # Store agent report
                self.agent_reports[msg.sender] = {
                    'content': msg.content,
                    'metadata': msg.metadata,
                    'step': msg.step,
                }
                
                # Update mental map if we have it
                if self.mental_map:
                    # Extract findings from message metadata
                    findings = msg.metadata.get('findings', {})
                    agent_pos = msg.metadata.get('position')
                    local_obs = msg.metadata.get('observation')
                    obs_radius = msg.metadata.get('observation_radius', 1)
                    current_step = msg.step
                    
                    f_count = len(findings.get('fires', []))
                    v_count = len(findings.get('victims', []))
                    b_count = len(findings.get('blocked_roads', []))
                    c_count = len(findings.get('collapsed_buildings', []))
                    if findings:
                        print(f"[Commander] Report from {msg.sender}: "
                              f"{f_count} fires, {v_count} victim-cells, "
                              f"{b_count} blocked, {c_count} collapsed")
                    
                    # First, update from full local observation (for scouts)
                    if agent_pos and local_obs:
                        self.mental_map.update_from_observation(
                            agent_pos, local_obs, obs_radius, current_step
                        )
                    
                    # Then, process exact coordinate findings
                    # Update fires with exact coordinates
                    for fire_x, fire_y, intensity in findings.get('fires', []):
                        pos = (fire_x, fire_y)
                        
                        if pos in self.mental_map.cells:
                            cell = self.mental_map.cells[pos]
                            cell.hazard = HazardType.FIRE
                            cell.fire_intensity = intensity
                            cell.explored = True
                            cell.last_updated_step = current_step
                    
                    # Update blocked roads with exact coordinates
                    for block_x, block_y in findings.get('blocked_roads', []):
                        pos = (block_x, block_y)
                        if pos in self.mental_map.cells:
                            cell = self.mental_map.cells[pos]
                            was_blocked = cell.blocked
                            cell.blocked = True
                            cell.explored = True
                            cell.last_updated_step = current_step
                            
                            # Update graph if newly blocked
                            if not was_blocked and pos in self.mental_map.graph:
                                self.mental_map.graph.remove_node(pos)
                    
                    # Update collapsed buildings with exact coordinates
                    for collapse_x, collapse_y in findings.get('collapsed_buildings', []):
                        pos = (collapse_x, collapse_y)

                        if pos in self.mental_map.cells:
                            cell = self.mental_map.cells[pos]
                            b_id = cell.building_id
                            if b_id is not None:
                                # Mark all cells of this building as having debris
                                for b_pos, b_cell in self.mental_map.cells.items():
                                    if b_cell.building_id == b_id:
                                        b_cell.hazard = HazardType.DEBRIS
                                        b_cell.explored = True
                                        b_cell.last_updated_step = current_step
                    
                    # If this message is a rescue confirmation, the agent has
                    # confirmed danger — trust victim findings unconditionally.
                    is_rescue_report = msg.metadata.get('rescued_at') is not None

                    # Update victim locations with exact coordinates
                    for victim_x, victim_y, count in findings.get('victims', []):
                        pos = (victim_x, victim_y)
                        if pos in self.mental_map.cells:
                            cell = self.mental_map.cells[pos]

                            # Determine if this cell's building is in danger
                            in_danger = is_rescue_report  # trust rescue reports
                            if not in_danger:
                                if cell.building_id is not None:
                                    b_id = cell.building_id
                                    building = self.mental_map.buildings.get(b_id)
                                    if building and building.collapsed:
                                        in_danger = True
                                    else:
                                        building_cells = [c for c in self.mental_map.cells.values() if c.building_id == b_id]
                                        if any(c.hazard in (HazardType.FIRE, HazardType.DEBRIS) for c in building_cells):
                                            in_danger = True
                                else:
                                    if cell.hazard in (HazardType.FIRE, HazardType.DEBRIS):
                                        in_danger = True

                            if in_danger:
                                # Keep rescued victims, replace unrescued with fresh count
                                rescued_victims = [v for v in cell.victims if v.rescued]
                                cell.victims = rescued_victims + [
                                    Victim(victim_id=-1, position=pos) for _ in range(count)
                                ]

                            cell.explored = True
                            cell.last_updated_step = current_step

                    # Process rescue confirmations — mark victims as rescued
                    rescued_at = msg.metadata.get('rescued_at')
                    if rescued_at:
                        rx, ry = rescued_at
                        # Mark victims as rescued in radius 1 (3x3, matches env rescue radius)
                        for dx in range(-1, 2):
                            for dy in range(-1, 2):
                                rpos = (rx + dx, ry + dy)
                                if rpos in self.mental_map.cells:
                                    rcell = self.mental_map.cells[rpos]
                                    for v in rcell.victims:
                                        if not v.rescued:
                                            v.rescued = True
                                    rcell.last_updated_step = current_step
                        print(f"[Commander] Mental map updated: rescue at ({rx},{ry})")

    def decide(self, observation: dict, messages: List[Message],
               env=None) -> List[Message]:
        """Produce task assignments. To be overridden."""
        raise NotImplementedError

    def _get_agent_type(self, agent_id: str) -> str:
        """Infer agent type from ID naming convention."""
        for t in ['scout', 'firefighter', 'medic']:
            if t in agent_id.lower():
                return t
        return 'unknown'
    
    def _get_free_agents(self) -> List[str]:
        """Return list of agents that are currently available (not busy with tasks).
        
        For medics/firefighters: directly inspect agent.current_task (no reports needed).
        For scouts: use report-based tracking (they send reports every step).
        """
        free_agents = []
        for agent_id in self.agent_ids:
            agent_obj = self.field_agents.get(agent_id)

            # --- Non-scout agents: direct object inspection ----------
            if agent_obj and agent_obj.agent_type != 'scout':
                if agent_obj.current_task is None:
                    # Clean up stale assignment ledger entry
                    self.assignments.pop(agent_id, None)
                    free_agents.append(agent_id)
                continue

            # --- Scout agents: report-based tracking -----------------
            if agent_id in self.assignments:
                assigned_step = self.assignments[agent_id].get('assigned_step', -1)
                report = self.agent_reports.get(agent_id, {})
                report_step = report.get('step', -1)
                metadata = report.get('metadata', {})

                # If the scout hasn't reported since being assigned, assume busy
                if report_step < assigned_step:
                    continue

                # Scout reported after assignment — check if it finished
                reported_task = metadata.get('current_task')
                if reported_task is not None:
                    continue
                else:
                    del self.assignments[agent_id]
                    free_agents.append(agent_id)
                    continue

            # No active assignment — check latest report
            report = self.agent_reports.get(agent_id, {})
            metadata = report.get('metadata', {})
            current_task = metadata.get('current_task')
            if current_task is not None:
                continue

            free_agents.append(agent_id)
        return free_agents  # May be empty — that is intentional
    
    def _zone_data_from_mental_map(self, zone_size: int = 10) -> List[dict]:
        """Build zone summaries from mental map instead of ground truth."""
        if not self.mental_map:
            return []
        
        zones = []
        for y in range(zone_size//2, self.mental_map.height - zone_size//2, zone_size):
            for x in range(zone_size//2, self.mental_map.width - zone_size//2, zone_size):
                zone_summary = self.mental_map.get_zone_summary(x, y, zone_size)
                # Rename keys for backward compatibility
                zone_summary['victims'] = zone_summary['victims_known']
                zone_summary['fires'] = zone_summary['fires_known']
                zones.append(zone_summary)
        return zones
    
    def _compute_path_on_mental_map(self, start: Tuple[int, int], goal: Tuple[int, int]) -> Optional[List[Tuple[int, int]]]:
        """Compute path using A* on mental map. Treats unknown cells with higher cost."""
        if not self.mental_map:
            return None
        
        import networkx as nx
        
        # Build weighted graph from mental map
        # Known passable cells: cost 1
        # Unknown cells: cost 2 (exploration penalty)
        # Blocked cells: not traversable (not in graph)
        weighted_graph = nx.Graph()
        
        for pos, cell in self.mental_map.cells.items():
            # Add node if it's potentially passable
            if cell.cell_type == CellType.ROAD:
                if cell.explored and cell.blocked:
                    continue  # Known blocked - skip
                # Add node with position
                weighted_graph.add_node(pos)
        
        # Add edges with costs
        for pos in weighted_graph.nodes():
            cell = self.mental_map.cells[pos]
            # Cost for this cell: 1 if explored, 2 if unexplored
            cost_here = 1.0 if cell.explored else 2.0
            
            for dx, dy in [(1, 0), (0, 1), (-1, 0), (0, -1)]:
                neighbor = (pos[0] + dx, pos[1] + dy)
                if neighbor in weighted_graph.nodes():
                    neighbor_cell = self.mental_map.cells[neighbor]
                    cost_neighbor = 1.0 if neighbor_cell.explored else 2.0
                    # Edge cost is average of both cells
                    edge_cost = (cost_here + cost_neighbor) / 2.0
                    weighted_graph.add_edge(pos, neighbor, weight=edge_cost)
        
        if start not in weighted_graph or goal not in weighted_graph:
            return None
        
        try:
            # A* with weighted edges
            path = nx.astar_path(
                weighted_graph, start, goal,
                heuristic=lambda a, b: abs(a[0]-b[0]) + abs(a[1]-b[1]),
                weight='weight'
            )
            return path
        except nx.NetworkXNoPath:
            pass

        # Fallback to BFS on unweighted graph
        try:
            return nx.shortest_path(self.mental_map.graph, start, goal)
        except nx.NetworkXNoPath:
            pass

        # Last resort: find the nearest reachable tile to the goal
        try:
            reachable = nx.node_connected_component(weighted_graph, start)
            best_node = min(
                reachable,
                key=lambda n: abs(n[0] - goal[0]) + abs(n[1] - goal[1])
            )
            if best_node != start:
                path = nx.astar_path(
                    weighted_graph, start, best_node,
                    heuristic=lambda a, b: abs(a[0]-b[0]) + abs(a[1]-b[1]),
                    weight='weight'
                )
                print(f"[Commander] Fallback: pathing to nearest reachable tile {best_node} "
                      f"(goal was {goal}, dist={abs(best_node[0]-goal[0])+abs(best_node[1]-goal[1])})")
                return path
        except (nx.NetworkXError, ValueError):
            pass

        return None
    
    def _attach_paths_to_commands(self, commands: List[Message], agent_positions: Dict[str, Tuple[int, int]]) -> List[Message]:
        """Compute paths for each command and attach to metadata."""
        for cmd in commands:
            agent_id = cmd.receiver
            target_pos = cmd.metadata.get('target_pos')
            if agent_id in agent_positions and target_pos:
                start_pos = agent_positions[agent_id]
                path = self._compute_path_on_mental_map(start_pos, target_pos)
                if path:
                    cmd.metadata['path'] = path
                    # Log path computation
                    print(f"[Commander] Computed path for {agent_id}: length {len(path)}")
                else:
                    print(f"[Commander] Warning: No path found for {agent_id} from {start_pos} to {target_pos}")
        return commands
    
    def _find_nearest_agent(self, target_pos: Tuple[int, int], 
                           agent_ids: List[str], 
                           agent_positions: Dict[str, Tuple[int, int]]) -> Optional[str]:
        """Find nearest agent to target position based on mental map distance."""
        if not agent_ids or not self.mental_map:
            return None
        
        nearest = None
        min_dist = float('inf')
        
        for agent_id in agent_ids:
            if agent_id not in agent_positions:
                continue
            pos = agent_positions[agent_id]
            # Use Manhattan distance as heuristic
            dist = abs(pos[0] - target_pos[0]) + abs(pos[1] - target_pos[1])
            if dist < min_dist:
                min_dist = dist
                nearest = agent_id
        
        return nearest


class HeuristicCommander(CommanderAgent):
    """
    Rule-based commander:
    1. Assign scouts to unexplored zones with most unknowns.
    2. Assign firefighters to zones with most fires.
    3. Assign medics to zones with most living victims.
    """

    def decide(self, observation: dict, messages: List[Message],
               env=None) -> List[Message]:
        self.step = observation.get('step', 0)
        # Use mental map zone data instead of ground truth
        if self.mental_map:
            self.zone_data = self._zone_data_from_mental_map(zone_size=10)
        else:
            self.zone_data = observation.get('zones', [])
        self.update_mental_map(messages)

        if self.mental_map:
            explored_pct = self.mental_map.get_explored_fraction() * 100
            known_victims = self.mental_map.get_all_known_victims()
            total_v = sum(v[2] for v in known_victims)
            print(f"\n{'='*60}")
            print(f"🗺️  [Commander] Step {self.step} | Mental Map: {explored_pct:.1f}% explored | "
                  f"{total_v} victims known | {len(self.zone_data)} zones")
            for z in self.zone_data:
                v = z.get('victims_known', z.get('victims', 0))
                f = z.get('fires_known', z.get('fires', 0))
                e = z.get('exploration', 0)
                if v > 0 or f > 0:
                    print(f"   Zone ({z['zone'][0]:2d},{z['zone'][1]:2d}): "
                          f"victims={v}, fires={f}, explored={e:.0%}")

        commands = []
        agent_positions = observation.get('agent_positions', {})

        # Get free agents only
        free_agents = self._get_free_agents()
        print(f"  👥 Free agents: {free_agents}")
        scouts = [a for a in free_agents if 'scout' in a.lower()]
        firefighters = [a for a in free_agents if 'firefighter' in a.lower()]
        medics = [a for a in free_agents if 'medic' in a.lower()]

        # Priority: zones with highest victim/fire counts
        unexplored_zones = sorted(self.zone_data, key=lambda z: z.get('exploration', 0))  # Least explored first

        # Split victims by hazard type:
        #   - Firefighters → buildings with FIRE + victims
        #   - Medics → buildings with DEBRIS (collapsed) + victims
        if self.mental_map:
            fire_victims, debris_victims = self.mental_map.get_known_victims_by_hazard()

            # 1. Assign firefighters to fire+victim locations (extinguish fire & rescue)
            fire_victims.sort(key=lambda v: v[2], reverse=True)
            for victim_x, victim_y, count in fire_victims:
                if not firefighters:
                    break
                nearest_ff = self._find_nearest_agent((victim_x, victim_y), firefighters, agent_positions)
                if nearest_ff:
                    cmd = make_task_assignment(
                        nearest_ff, 'extinguish_fire', (victim_x, victim_y), self.step,
                        f"Fire + {count} victims at ({victim_x},{victim_y}). Extinguish and rescue."
                    )
                    commands.append(cmd)
                    self.assignments[nearest_ff] = {'task': 'extinguish', 'target': (victim_x, victim_y), 'assigned_step': self.step}
                    firefighters.remove(nearest_ff)

            # Also send remaining firefighters to fires without victims
            known_fires = self.mental_map.get_all_known_fires()
            fire_victim_positions = {(v[0], v[1]) for v in fire_victims}
            fires_only = [(fx, fy, intensity) for fx, fy, intensity in known_fires
                          if (fx, fy) not in fire_victim_positions]
            fires_only.sort(key=lambda f: f[2], reverse=True)
            for fire_x, fire_y, intensity in fires_only:
                if not firefighters:
                    break
                nearest_ff = self._find_nearest_agent((fire_x, fire_y), firefighters, agent_positions)
                if nearest_ff:
                    cmd = make_task_assignment(
                        nearest_ff, 'extinguish_fire', (fire_x, fire_y), self.step,
                        f"Fire (intensity={intensity:.1f}) at ({fire_x},{fire_y})."
                    )
                    commands.append(cmd)
                    self.assignments[nearest_ff] = {'task': 'extinguish', 'target': (fire_x, fire_y), 'assigned_step': self.step}
                    firefighters.remove(nearest_ff)

            # 2. Assign medics to collapsed+victim locations (rescue from debris)
            debris_victims.sort(key=lambda v: v[2], reverse=True)
            for victim_x, victim_y, count in debris_victims:
                if not medics:
                    break
                nearest_medic = self._find_nearest_agent((victim_x, victim_y), medics, agent_positions)
                if nearest_medic:
                    cmd = make_task_assignment(
                        nearest_medic, 'rescue_victims', (victim_x, victim_y), self.step,
                        f"Collapsed building + {count} victims at ({victim_x},{victim_y})."
                    )
                    commands.append(cmd)
                    self.assignments[nearest_medic] = {'task': 'rescue', 'target': (victim_x, victim_y), 'assigned_step': self.step}
                    medics.remove(nearest_medic)

            # Also send remaining medics to fire+victim locations if firefighters exhausted
            for victim_x, victim_y, count in fire_victims:
                if not medics:
                    break
                nearest_medic = self._find_nearest_agent((victim_x, victim_y), medics, agent_positions)
                if nearest_medic:
                    cmd = make_task_assignment(
                        nearest_medic, 'rescue_victims', (victim_x, victim_y), self.step,
                        f"Fire + {count} victims at ({victim_x},{victim_y}). Rescue assist."
                    )
                    commands.append(cmd)
                    self.assignments[nearest_medic] = {'task': 'rescue', 'target': (victim_x, victim_y), 'assigned_step': self.step}
                    medics.remove(nearest_medic)

        # 3. Assign scouts to unexplored zones
        for zone in unexplored_zones:
            if not scouts:
                break
            zx, zy = zone['zone']
            zone_size = 10
            zone_bounds = (zx - zone_size // 2, zy - zone_size // 2,
                           zx + zone_size // 2, zy + zone_size // 2)  # (x_min, y_min, x_max, y_max)
            target = (zx, zy)  # zone center
            nearest_scout = self._find_nearest_agent(target, scouts, agent_positions)
            if nearest_scout:
                cmd = make_task_assignment(
                    nearest_scout, 'search_zone', target, self.step,
                    f"Explore zone ({zone.get('exploration', 0):.1%} explored).",
                    zone_bounds=zone_bounds,
                )
                commands.append(cmd)
                self.assignments[nearest_scout] = {'task': 'scout', 'zone': zone['zone'], 'assigned_step': self.step}
                scouts.remove(nearest_scout)

        # Attach paths to all commands
        commands = self._attach_paths_to_commands(commands, agent_positions)

        for cmd in commands:
            path_len = len(cmd.metadata.get('path', []))
            print(f"[Commander] → {cmd.receiver}: {cmd.metadata.get('task_type')} "
                  f"at {cmd.metadata.get('target_pos')} (path={path_len} steps)")
        print(f"{'='*60}\n")

        return commands

    def replan(self, event: dict, observation: dict,
               messages: List[Message]) -> List[Message]:
        """Re-plan in response to a black swan or aftershock event."""
        self.replan_log.append({'step': self.step, 'event': event})
        # Simply re-run decide with updated info
        return self.decide(observation, messages)


class LLMCommander(CommanderAgent):
    """
    LLM-powered commander. Uses an API call to generate strategic assignments.
    Falls back to HeuristicCommander if LLM call fails.
    """

    def __init__(self, agent_ids: List[str],
                 api_key: Optional[str] = None,
                 model: str = 'gpt-4o-mini',
                 provider: str = 'openai'):
        super().__init__(agent_ids)
        self.api_key = "sk-pO9q3QH-PUbg1khdeFPN6Q"
        # self.api_key = ""
        self.model = model
        self.provider = provider
        # self.fallback = HeuristicCommander(agent_ids)
        self.llm_call_count = 0
        self.total_tokens = 0
        self.call_log: List[dict] = []  # per-call latency and token tracking

    def _build_prompt(self, observation: dict, messages: List[Message]) -> str:
        """Build the LLM prompt from observation and messages."""
        # Use ground-truth zone summaries (noisy but from the actual environment)
        zones_summary = ""
        for z in observation.get('zones', []):
            zones_summary += (
                f"  Zone ({z['zone'][0]},{z['zone'][1]}): "
                f"victims={z.get('victims', z.get('victims_alive', 0))}, "
                f"fires={z.get('fires', 0)}, "
                f"blocked={z.get('blocked_roads', 0)}, collapsed={z.get('collapsed_buildings', 0)}\n"
            )

        agent_reports = ""
        for msg in messages:
            agent_reports += f"  {msg.to_semantic()}\n"

        agent_positions = observation.get('agent_positions', {})
        free_agents = self._get_free_agents()
        if not free_agents:
            print("No free agents available for new tasks. Skipping LLM prompt.")
            return ""
        agents_info = ""
        for aid in free_agents:
            if aid in agent_positions:
                pos = agent_positions[aid]
                atype = self._get_agent_type(aid)
                agents_info += f"  {aid} ({atype}) at ({pos[0]},{pos[1]}) - AVAILABLE\n"

        # Collect exact known victim and fire locations from mental map
        fire_victim_str = ""
        debris_victim_str = ""
        fire_locations_str = ""
        if self.mental_map:
            fire_victims, debris_victims = self.mental_map.get_known_victims_by_hazard()
            if fire_victims:
                fire_victims.sort(key=lambda v: v[2], reverse=True)
                for vx, vy, count in fire_victims:
                    fire_victim_str += f"  ({vx},{vy}): {count} victim(s) in FIRE\n"
            if debris_victims:
                debris_victims.sort(key=lambda v: v[2], reverse=True)
                for vx, vy, count in debris_victims:
                    debris_victim_str += f"  ({vx},{vy}): {count} victim(s) in COLLAPSED building\n"
            known_fires = self.mental_map.get_all_known_fires()
            if known_fires:
                known_fires.sort(key=lambda f: f[2], reverse=True)
                for fx, fy, intensity in known_fires:
                    fire_locations_str += f"  ({fx},{fy}): intensity={intensity:.1f}\n"

        prompt = f"""You are the Commander of an urban disaster response operation after an earthquake.
Your role is to assign tasks to field agents to maximize civilian survival.

CURRENT STEP: {observation.get('step', 0)}

ZONE SUMMARIES (noisy estimates from environment):
{zones_summary}

VICTIMS IN BURNING BUILDINGS (send firefighters here - extinguish & rescue):
{fire_victim_str if fire_victim_str else "  None discovered yet."}

VICTIMS IN COLLAPSED BUILDINGS (send medics here - rescue from debris):
{debris_victim_str if debris_victim_str else "  None discovered yet."}

FIRE LOCATIONS WITHOUT KNOWN VICTIMS (send firefighters to contain):
{fire_locations_str if fire_locations_str else "  None discovered yet."}

FREE AGENTS AVAILABLE FOR NEW TASKS:
{agents_info if agents_info else "  No free agents available."}

RECENT AGENT REPORTS:
{agent_reports if agent_reports else "  No reports this step."}

AGENT TYPES AND CAPABILITIES:
- scout_*: Can scan/explore areas, fast movement, observation radius=2
- firefighter_*: Can extinguish fires AND rescue victims from burning buildings. Send to cells with FIRE + victims.
- medic_*: Can rescue victims from collapsed/debris buildings. Send to cells with COLLAPSED buildings + victims.

Respond with a JSON array of task assignments. Each assignment:
{{
  "agent_id": "<agent_id>",
  "task_type": "search_zone|rescue_victims|extinguish_fire|move_to",
  "target_x": <int>,
  "target_y": <int>,
  "reason": "<brief reason>"
}}

PRIORITIES:
1. Assign firefighters to burning buildings with victims (extinguish_fire task) — they will extinguish and rescue
2. Assign medics to collapsed buildings with victims (rescue_victims task) — they rescue from debris
3. Assign remaining firefighters to fires without victims
4. Assign remaining medics to any remaining victim locations
5. Explore unexplored zones with high uncertainty (assign scouts)
6. Consider blocked roads - paths will be planned around them

IMPORTANT: Only assign FREE agents listed above. Busy agents cannot accept new tasks.

Respond ONLY with the JSON array, no other text."""

        return prompt

    def decide(self, observation: dict, messages: List[Message],
               env=None) -> List[Message]:
        self.step = observation.get('step', 0)
        self.zone_data = observation.get('zones', [])
        
        self.update_mental_map(messages)
        agent_positions = observation.get('agent_positions', {})

        # Pass the original observation (ground-truth zones) to the prompt
        prompt = self._build_prompt(observation, messages)

        print(f"\n{'='*60}")
        print(f"[LLM Commander] Step {self.step} | Prompt sent to LLM:")
        print(f"{'-'*60}")
        print(prompt)
        print(f"{'-'*60}")

        # try:
        if not prompt:
            print("[LLM Commander] No prompt generated (no free agents). Skipping LLM call.")
            return []

        assignments = self._call_llm(prompt)

        print(f"[LLM Commander] Raw LLM response:")
        print(f"{'-'*60}")
        print(assignments)
        print(f"{'-'*60}")

        commands = self._parse_assignments(assignments)

        # Snap zone-level LLM targets to exact victim/fire coordinates
        if commands:
            commands = [self._refine_target(cmd, agent_positions) for cmd in commands]

        # Attach paths to all commands
        if commands:
            commands = self._attach_paths_to_commands(commands, agent_positions)
        
        print(f"[LLM Commander] Step {self.step} -> {len(commands)} task(s) assigned")
        for cmd in commands:
            path_len = len(cmd.metadata.get('path', []))
            print(f"  >> {cmd.receiver}: {cmd.metadata.get('task_type')} "
                  f"at {cmd.metadata.get('target_pos')} (path={path_len} steps)")
        print(f"{'='*60}\n")

        return commands if commands else []

    # def _call_llm(self, prompt: str) -> str:
    #     """Call the LLM API. Override for different providers."""
    #     self.llm_call_count += 1

    #     if self.provider == 'openai' and self.api_key:
    #         import openai
    #         client = openai.OpenAI(api_key=self.api_key)
    #         response = client.chat.completions.create(
    #             model=self.model,
    #             messages=[
    #                 {"role": "system", "content": "You are a disaster response commander. Respond only with JSON."},
    #                 {"role": "user", "content": prompt},
    #             ],
    #             max_tokens=500,
    #             temperature=0.3,
    #         )
    #         self.total_tokens += response.usage.total_tokens if response.usage else 0
    #         return response.choices[0].message.content

    #     elif self.provider == 'anthropic' and self.api_key:
    #         import anthropic
    #         client = anthropic.Anthropic(api_key=self.api_key)
    #         response = client.messages.create(
    #             model=self.model,
    #             max_tokens=500,
    #             messages=[{"role": "user", "content": prompt}],
    #             system="You are a disaster response commander. Respond only with JSON.",
    #         )
    #         return response.content[0].text

    #     else:
    #         # Simulate LLM response for testing (uses heuristic logic)
    #         return self._simulated_llm_response()
    def _call_llm(self, prompt: str) -> str:
        """Call Triton API via OpenAI client."""
        import time
        from openai import OpenAI

        self.llm_call_count += 1

        client = OpenAI(
            timeout=120.0,
            api_key=self.api_key,
            base_url="https://tritonai-api.ucsd.edu",
        )

        start_time = time.time()
        response = client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are a disaster response commander. Respond only with JSON."},
                {"role": "user", "content": prompt},
            ],
            max_tokens=8192,
            temperature=0.3,
        )
        latency = time.time() - start_time

        usage = response.usage
        input_tokens = usage.prompt_tokens if usage else 0
        output_tokens = usage.completion_tokens if usage else 0
        total = usage.total_tokens if usage else 0
        self.total_tokens += total

        self.call_log.append({
            'step': self.step,
            'latency_seconds': round(latency, 3),
            'input_tokens': input_tokens,
            'output_tokens': output_tokens,
            'total_tokens': total,
        })

        return response.choices[0].message.content

    def _simulated_llm_response(self) -> str:
        """Generate a simulated LLM response based on heuristic logic (for testing without API)."""
        assignments = []

        # Simple heuristic: assign based on zone priorities
        fire_zones = sorted(self.zone_data, key=lambda z: z.get('fires', 0), reverse=True)
        victim_zones = sorted(self.zone_data, key=lambda z: z.get('victims_alive', 0), reverse=True)

        used_agents = set()
        # Medics to victims
        for aid in self.agent_ids:
            if 'medic' in aid and victim_zones:
                z = victim_zones.pop(0) if victim_zones[0].get('victims_alive', 0) > 0 else None
                if z:
                    assignments.append({
                        'agent_id': aid,
                        'task_type': 'rescue_victims',
                        'target_x': z['zone'][0] + 5,
                        'target_y': z['zone'][1] + 5,
                        'reason': f"{z['victims_alive']} victims alive"
                    })
                    used_agents.add(aid)

        # Firefighters to fires
        for aid in self.agent_ids:
            if 'firefighter' in aid and fire_zones:
                z = fire_zones.pop(0) if fire_zones[0].get('fires', 0) > 0 else None
                if z:
                    assignments.append({
                        'agent_id': aid,
                        'task_type': 'extinguish_fire',
                        'target_x': z['zone'][0] + 5,
                        'target_y': z['zone'][1] + 5,
                        'reason': f"{z['fires']} fires"
                    })
                    used_agents.add(aid)

        # Scouts explore
        zone_idx = 0
        for aid in self.agent_ids:
            if aid not in used_agents and self.zone_data:
                z = self.zone_data[zone_idx % len(self.zone_data)]
                zx, zy = z['zone']
                assignments.append({
                    'agent_id': aid,
                    'task_type': 'search_zone',
                    'target_x': zx,
                    'target_y': zy,
                    'reason': 'explore area'
                })
                zone_idx += 1

        return json.dumps(assignments)

    def _parse_assignments(self, response_text: str) -> List[Message]:
        """Parse LLM JSON response into task assignment messages."""
        # Clean up response
        text = response_text.strip()
        # Try to extract JSON array
        match = re.search(r'\[.*\]', text, re.DOTALL)
        if match:
            text = match.group()

        try:
            assignments = json.loads(text)
        except json.JSONDecodeError:
            return []

        # Get free agents to filter assignments
        free_agents = self._get_free_agents()
        
        commands = []
        for a in assignments:
            agent_id = a.get('agent_id', '')
            if agent_id not in self.agent_ids or agent_id not in free_agents:
                continue  # Skip if not valid or not free
            task_type = a.get('task_type', 'move_to')
            tx = a.get('target_x', 25)
            ty = a.get('target_y', 25)
            reason = a.get('reason', '')

            # For search_zone tasks, compute and attach zone_bounds
            extra = {}
            if task_type == 'search_zone':
                zone_size = 10
                extra['zone_bounds'] = (tx - zone_size // 2, ty - zone_size // 2,
                                        tx + zone_size // 2, ty + zone_size // 2)

            cmd = make_task_assignment(
                agent_id, task_type, (tx, ty), self.step, reason, **extra
            )
            commands.append(cmd)
            self.assignments[agent_id] = {
                'task': task_type, 'target': (tx, ty), 'reason': reason, 'assigned_step': self.step
            }

        return commands

    def _refine_target(self, cmd: Message, agent_positions: Dict[str, Tuple[int, int]]) -> Message:
        """Snap LLM zone-level target to nearest actual victim/fire from mental map.
        Firefighters → fire+victim cells.  Medics → debris+victim cells.
        """
        if not self.mental_map:
            return cmd

        task_type = cmd.metadata.get('task_type', '')
        target_pos = cmd.metadata.get('target_pos')
        if not target_pos:
            return cmd

        tx, ty = target_pos
        fire_victims, debris_victims = self.mental_map.get_known_victims_by_hazard()

        if task_type == 'rescue_victims':
            # Medics → prefer debris/collapsed victims
            candidates = debris_victims if debris_victims else fire_victims
            if candidates:
                best = min(
                    candidates,
                    key=lambda v: (abs(v[0] - tx) + abs(v[1] - ty), -v[2])
                )
                new_pos = (best[0], best[1])
                if new_pos != target_pos:
                    print(f"[LLM Commander] Refined {cmd.receiver} rescue target "
                          f"({tx},{ty}) -> ({new_pos[0]},{new_pos[1]}) "
                          f"({best[2]} victims)")
                    cmd.metadata['target_pos'] = new_pos

        elif task_type == 'extinguish_fire':
            # Firefighters → prefer fire+victim cells, then any fire
            if fire_victims:
                best = min(
                    fire_victims,
                    key=lambda v: (abs(v[0] - tx) + abs(v[1] - ty), -v[2])
                )
                new_pos = (best[0], best[1])
                if new_pos != target_pos:
                    print(f"[LLM Commander] Refined {cmd.receiver} fire+victim target "
                          f"({tx},{ty}) -> ({new_pos[0]},{new_pos[1]}) "
                          f"({best[2]} victims)")
                    cmd.metadata['target_pos'] = new_pos
            else:
                known_fires = self.mental_map.get_all_known_fires()
                if known_fires:
                    best = min(
                        known_fires,
                        key=lambda f: (abs(f[0] - tx) + abs(f[1] - ty), -f[2])
                    )
                    new_pos = (best[0], best[1])
                    if new_pos != target_pos:
                        print(f"[LLM Commander] Refined {cmd.receiver} fire target "
                              f"({tx},{ty}) -> ({new_pos[0]},{new_pos[1]}) "
                              f"(intensity={best[2]:.1f})")
                        cmd.metadata['target_pos'] = new_pos

        return cmd

    def get_stats(self) -> dict:
        return {
            'llm_calls': self.llm_call_count,
            'total_tokens': self.total_tokens,
            'call_log': self.call_log,
            'current_assignments': dict(self.assignments),
        }
