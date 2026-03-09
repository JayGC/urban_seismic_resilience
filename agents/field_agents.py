"""
Field agents: Scout, Firefighter, Medic.
Each uses heuristic policies (greedy nearest target, A* pathfinding).
"""

import numpy as np
from typing import Optional, Dict, List, Tuple, Any
from .messages import Message, MessageBus, MessageType, make_report, make_emergency


class FieldAgent:
    """Base class for field agents with local observation and heuristic movement."""

    def __init__(self, agent_id: str, agent_type: str, position: Tuple[int, int],
                 observation_radius: int = 1):
        self.agent_id = agent_id
        self.agent_type = agent_type
        self.position = position
        self.observation_radius = observation_radius
        self.current_task: Optional[dict] = None
        self.path: List[Tuple[int, int]] = []
        self.idle_steps = 0
        self.total_steps = 0
        self.actions_taken: List[str] = []
        self.status = 'idle'

    def observe(self, env) -> dict:
        """Get local observation from environment."""
        obs = env.get_agent_observation(self.agent_id, self.observation_radius)
        self.position = env.agent_positions.get(self.agent_id, self.position)
        return obs

    def decide(self, obs: dict, messages: List[Message], env) -> Tuple[dict, List[Message]]:
        """
        Decide action based on observation and received messages.
        Returns: (action_dict, outgoing_messages)
        """
        self.total_steps += 1
        outgoing = []

        # Check for task assignments — only accept if not already busy
        for msg in messages:
            if msg.msg_type == MessageType.TASK_ASSIGNMENT:
                if self.current_task is not None:
                    print(f"  XX [{self.agent_id}] Rejected task (busy with '{self.current_task['type']}'): {msg.content}")
                else:
                    print(f"  << [{self.agent_id}] Received task from commander: {msg.content}")
                    self._accept_task(msg, env)

        # If we have a task with a target, pathfind toward it
        if self.current_task and 'target_pos' in self.current_task:
            action = self._follow_task(env)
        else:
            action = self._autonomous_action(obs, env)

        # Only scouts generate reports (for commander mental-map updates).
        # Medics and firefighters focus solely on their tasks.
        if self.agent_type == 'scout':
            report = self._make_report(obs)
            outgoing.append(report)
            findings = report.metadata.get('findings', {})
            f_count = len(findings.get('fires', []))
            v_count = sum(v[2] for v in findings.get('victims', []))
            b_count = len(findings.get('blocked_roads', []))
            if f_count or v_count or b_count:
                print(f"  >> [{self.agent_id}] Report -> commander: "
                      f"{f_count} fires, {v_count} victims, {b_count} blocked roads")

        if action['type'] == 'noop':
            self.idle_steps += 1
        self.actions_taken.append(action['type'])
        self.status = action['type']
        print(f"  .. [{self.agent_id}] pos={self.position} action={action['type']} "
              f"task={'active: '+self.current_task['type'] if self.current_task else 'none'} "
              f"path_len={len(self.path)}")

        return action, outgoing

    def _accept_task(self, msg: Message, env):
        """Accept a task assignment from commander."""
        target_pos = msg.metadata.get('target_pos')
        task_type = msg.metadata.get('task_type', 'move_to')
        provided_path = msg.metadata.get('path', [])
        self.current_task = {
            'type': task_type,
            'target_pos': target_pos,
        }
        # Prefer commander-provided path (planned on mental map).
        if provided_path:
            self.path = [tuple(p) for p in provided_path]
            # Ensure path starts at current position for consistent following.
            if self.path and self.path[0] != self.position:
                self.path = [self.position] + self.path
            print(f"  OK [{self.agent_id}] Accepted task '{task_type}' -> {target_pos} "
                  f"with commander path (len={len(self.path)})")
        elif target_pos:
            # Fallback: local replanning if no path was provided.
            self.path = env.grid.shortest_path(self.position, target_pos) or []
            if not self.path:
                # No direct path — find nearest reachable tile to target
                self.path = self._path_to_nearest_reachable(env, target_pos)
            print(f"  !! [{self.agent_id}] Accepted task '{task_type}' -> {target_pos} "
                  f"with LOCAL fallback path (len={len(self.path)})")
        else:
            self.path = []
            print(f"  XX [{self.agent_id}] Accepted task '{task_type}' but no target/path")

    def _follow_task(self, env) -> dict:
        """Follow current task path."""
        if not self.path or len(self.path) <= 1:
            # Arrived at target or path broken
            action = self._task_action_at_target()
            self.current_task = None
            self.path = []
            return action

        # Check if next step is passable
        next_pos = self.path[1]
        if next_pos in env.grid.cells:
            next_cell = env.grid.cells[next_pos]
            # If we encounter an unexpected obstacle (blocked road)
            if next_cell.cell_type.name == 'ROAD' and next_cell.blocked:
                # Agent cannot see beyond its observation radius — abandon task
                # so the commander can reassign with updated mental map info.
                print(f"  XX [{self.agent_id}] Obstacle at {next_pos}, abandoning task")
                self.current_task = None
                self.path = []
                return {'type': 'noop'}
                return {'type': 'noop'}

        # Move along path
        dx = next_pos[0] - self.position[0]
        dy = next_pos[1] - self.position[1]
        self.path = self.path[1:]
        return {'type': 'move', 'dx': dx, 'dy': dy}

    def _task_action_at_target(self) -> dict:
        """Action to perform when arrived at task target."""
        if self.current_task:
            task_type = self.current_task.get('type', '')
            if 'search' in task_type or 'scan' in task_type:
                return {'type': 'scan'}
            elif 'rescue' in task_type or 'victim' in task_type:
                return {'type': 'rescue'}
            elif 'fire' in task_type or 'extinguish' in task_type:
                return {'type': 'extinguish'}
            elif 'treat' in task_type or 'medic' in task_type:
                return {'type': 'treat'}
        return {'type': 'scan'}

    def _autonomous_action(self, obs: dict, env) -> dict:
        """Default autonomous behavior (to be overridden by subclasses)."""
        return {'type': 'noop'}

    def _make_report(self, obs: dict) -> Message:
        """Generate a detailed report message with exact coordinates of findings."""
        local = obs.get('local_grid', [])
        
        # Collect exact coordinates of findings
        fires = []  # List of fire positions
        victims = []  # List of victim positions
        blockages = []  # List of blocked road positions
        collapses = []  # List of collapsed building positions
        
        radius = self.observation_radius
        agent_x, agent_y = self.position
        
        # Parse local observation grid with exact coordinates
        for dy_idx, row in enumerate(local):
            for dx_idx, cell in enumerate(row):
                if isinstance(cell, dict) and cell.get('type') != 'OUT_OF_BOUNDS':
                    # Convert observation indices to absolute coordinates
                    dx = dx_idx - radius
                    dy = dy_idx - radius
                    abs_x = agent_x + dx
                    abs_y = agent_y + dy
                    
                    # Record exact coordinates of hazards/blockages
                    if cell.get('hazard') == 'FIRE':
                        fires.append((abs_x, abs_y, cell.get('fire_intensity', 0)))
                    
                    if cell.get('num_victims', 0) > 0:
                        victims.append((abs_x, abs_y, cell.get('num_victims', 0)))
                    
                    if cell.get('blocked', False) and cell.get('type') == 'ROAD':
                        blockages.append((abs_x, abs_y))
                    
                    # Check for collapsed buildings (inferred from structure)
                    if cell.get('type') == 'BUILDING' and cell.get('collapsed', False):
                        collapses.append((abs_x, abs_y))
        
        # Build detailed metadata with exact coordinates
        report_metadata = {
            'position': self.position,
            'observation': local,
            'observation_radius': self.observation_radius,
            'findings': {
                'fires': fires,  # [(x, y, intensity), ...]
                'victims': victims,  # [(x, y, count), ...]
                'blocked_roads': blockages,  # [(x, y), ...]
                'collapsed_buildings': collapses,  # [(x, y), ...]
            },
            'summary': {
                'num_victims_nearby': sum(v[2] for v in victims),
                'fires_nearby': len(fires),
                'blocked_nearby': len(blockages),
                'collapsed_nearby': len(collapses),
            },
            # Add agent status for availability tracking
            'status': self.status,
            'current_task': self.current_task,
        }
        
        return make_report(
            self.agent_id, self.position,
            report_metadata,
            obs.get('step', 0)
        )

    def _find_nearest_target(self, env, target_type: str) -> Optional[Tuple[int, int]]:
        """Find nearest cell matching target type using BFS on the grid graph."""
        import networkx as nx
        if self.position not in env.grid.graph:
            return None

        # BFS from current position
        visited = set()
        queue = [self.position]
        visited.add(self.position)

        while queue:
            current = queue.pop(0)
            # Check cells within radius 2 (to find victims in nearby buildings)
            for dx in range(-2, 3):
                for dy in range(-2, 3):
                    check = (current[0] + dx, current[1] + dy)
                    if check in env.grid.cells:
                        cell = env.grid.cells[check]
                        if target_type == 'victim':
                            if any(not v.rescued and v.health > 0 for v in cell.victims):
                                return current  # Return the road cell nearby
                        elif target_type == 'fire':
                            if cell.hazard.name == 'FIRE':
                                return current

            for neighbor in env.grid.graph.neighbors(current):
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)

        return None

    def _move_toward(self, target: Tuple[int, int], env) -> dict:
        """Move one step toward target using pathfinding."""
        path = env.grid.shortest_path(self.position, target)
        if path and len(path) > 1:
            next_pos = path[1]
            dx = next_pos[0] - self.position[0]
            dy = next_pos[1] - self.position[1]
            return {'type': 'move', 'dx': dx, 'dy': dy}
        return {'type': 'noop'}

    def _path_to_nearest_reachable(self, env, target: Tuple[int, int]) -> List[Tuple[int, int]]:
        """Find the nearest reachable tile to target and return path to it."""
        import networkx as nx
        if self.position not in env.grid.graph:
            return []
        try:
            reachable = nx.node_connected_component(env.grid.graph, self.position)
            best = min(
                reachable,
                key=lambda n: abs(n[0] - target[0]) + abs(n[1] - target[1])
            )
            if best == self.position:
                return []
            path = env.grid.shortest_path(self.position, best) or []
            if path:
                print(f"  ~~ [{self.agent_id}] Fallback: nearest reachable to {target} is {best} "
                      f"(dist={abs(best[0]-target[0])+abs(best[1]-target[1])})")
            return path
        except (nx.NetworkXError, ValueError):
            return []

    def _random_move(self, env) -> dict:
        """Move in a random valid direction."""
        import random
        directions = [(1, 0), (-1, 0), (0, 1), (0, -1)]
        random.shuffle(directions)
        for dx, dy in directions:
            new_pos = (self.position[0] + dx, self.position[1] + dy)
            if new_pos in env.grid.cells and env.grid.cells[new_pos].passable:
                return {'type': 'move', 'dx': dx, 'dy': dy}
        return {'type': 'noop'}


class ScoutAgent(FieldAgent):
    """Scout: explores the map, scans for victims and hazards, reports findings."""

    def __init__(self, agent_id: str, position: Tuple[int, int]):
        super().__init__(agent_id, 'scout', position, observation_radius=2)
        self.zone_bounds: Optional[Tuple[int, int, int, int]] = None  # (x_min, y_min, x_max, y_max)

    def _accept_task(self, msg: Message, env):
        """Accept task — for search_zone tasks, store zone bounds."""
        # Store zone bounds before calling parent _accept_task
        self.zone_bounds = msg.metadata.get('zone_bounds')
        super()._accept_task(msg, env)
        if self.zone_bounds:
            x0, y0, x1, y1 = self.zone_bounds
            print(f"  OK [{self.agent_id}] Zone bounds set: ({x0},{y0})-({x1},{y1})")

    def _follow_task(self, env) -> dict:
        """Override: for search_zone tasks, systematically explore the entire zone."""
        # For non-zone tasks, use default behaviour
        if not self.zone_bounds or (self.current_task and
                self.current_task.get('type', '') != 'search_zone'):
            return super()._follow_task(env)

        # --- Zone exploration logic ---
        # If we still have a path, keep following it
        if self.path and len(self.path) > 1:
            next_pos = self.path[1]
            if next_pos in env.grid.cells:
                next_cell = env.grid.cells[next_pos]
                if next_cell.cell_type.name == 'ROAD' and next_cell.blocked:
                    print(f"  XX [{self.agent_id}] Obstacle at {next_pos} during zone sweep, replanning")
                    self.path = []
                    # Fall through to find next unexplored cell
                else:
                    dx = next_pos[0] - self.position[0]
                    dy = next_pos[1] - self.position[1]
                    self.path = self.path[1:]
                    return {'type': 'move', 'dx': dx, 'dy': dy}

        # Path exhausted — find next unexplored road cell in the zone
        next_target = self._get_next_unexplored_in_zone(env)
        if next_target:
            self.path = env.grid.shortest_path(self.position, next_target) or []
            if not self.path:
                self.path = self._path_to_nearest_reachable(env, next_target)
            if self.path and len(self.path) > 1:
                next_pos = self.path[1]
                dx = next_pos[0] - self.position[0]
                dy = next_pos[1] - self.position[1]
                self.path = self.path[1:]
                return {'type': 'move', 'dx': dx, 'dy': dy}
            # Can't reach this cell — skip it and scan in place
            return {'type': 'scan'}

        # All reachable road cells in zone explored — task complete
        x0, y0, x1, y1 = self.zone_bounds
        print(f"  OK [{self.agent_id}] Zone ({x0},{y0})-({x1},{y1}) fully explored, task complete")
        self.current_task = None
        self.path = []
        self.zone_bounds = None
        return {'type': 'scan'}

    def _get_next_unexplored_in_zone(self, env) -> Optional[Tuple[int, int]]:
        """BFS from current position to find nearest unexplored road cell within zone bounds."""
        if not self.zone_bounds:
            return None
        x_min, y_min, x_max, y_max = self.zone_bounds

        if self.position not in env.grid.graph:
            return None
        visited = set()
        queue = [self.position]
        visited.add(self.position)

        while queue:
            current = queue.pop(0)
            # Check if this cell is in the zone and unexplored
            cx, cy = current
            if (x_min <= cx < x_max and y_min <= cy < y_max
                    and not env.grid.cells[current].explored):
                return current
            for neighbor in env.grid.graph.neighbors(current):
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)
        return None

    def _autonomous_action(self, obs: dict, env) -> dict:
        """When no zone task is active, explore nearest unexplored cell globally."""
        unexplored = self._find_nearest_unexplored(env)
        if unexplored:
            return self._move_toward(unexplored, env)
        if not env.grid.cells.get(self.position, None) or \
           not env.grid.cells[self.position].explored:
            return {'type': 'scan'}
        return self._random_move(env)

    def _find_nearest_unexplored(self, env) -> Optional[Tuple[int, int]]:
        """BFS for nearest unexplored road cell (global, no zone restriction)."""
        if self.position not in env.grid.graph:
            return None
        visited = set()
        queue = [self.position]
        visited.add(self.position)
        while queue:
            current = queue.pop(0)
            if not env.grid.cells[current].explored:
                return current
            for neighbor in env.grid.graph.neighbors(current):
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)
        return None


class FirefighterAgent(FieldAgent):
    """Firefighter: moves toward fires, extinguishes them."""

    def __init__(self, agent_id: str, position: Tuple[int, int]):
        super().__init__(agent_id, 'firefighter', position)

    def _autonomous_action(self, obs: dict, env) -> dict:
        # Check if adjacent to fire
        local = obs.get('local_grid', [])
        for row in local:
            for cell in row:
                if isinstance(cell, dict) and cell.get('hazard') == 'FIRE':
                    return {'type': 'extinguish'}

        # Find nearest fire
        target = self._find_nearest_target(env, 'fire')
        if target:
            return self._move_toward(target, env)

        # No fires, help with victims
        target = self._find_nearest_target(env, 'victim')
        if target:
            return self._move_toward(target, env)

        return self._random_move(env)


class MedicAgent(FieldAgent):
    """Medic: moves toward victims, rescues and treats them."""

    def __init__(self, agent_id: str, position: Tuple[int, int]):
        super().__init__(agent_id, 'medic', position)

    def _autonomous_action(self, obs: dict, env) -> dict:
        # Check if adjacent to victims
        local = obs.get('local_grid', [])
        has_victims_nearby = False
        for row in local:
            for cell in row:
                if isinstance(cell, dict) and cell.get('num_victims', 0) > 0:
                    has_victims_nearby = True
                    break

        if has_victims_nearby:
            return {'type': 'rescue'}

        # Find nearest victim
        target = self._find_nearest_target(env, 'victim')
        if target:
            return self._move_toward(target, env)

        return self._random_move(env)
