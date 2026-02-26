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

        # Check for task assignments
        for msg in messages:
            if msg.msg_type == MessageType.TASK_ASSIGNMENT:
                self._accept_task(msg, env)

        # If we have a task with a target, pathfind toward it
        if self.current_task and 'target_pos' in self.current_task:
            action = self._follow_task(env)
        else:
            action = self._autonomous_action(obs, env)

        # Send periodic reports
        if self.total_steps % 3 == 0:
            report = self._make_report(obs)
            outgoing.append(report)

        if action['type'] == 'noop':
            self.idle_steps += 1
        self.actions_taken.append(action['type'])
        self.status = action['type']

        return action, outgoing

    def _accept_task(self, msg: Message, env):
        """Accept a task assignment from commander."""
        target_pos = msg.metadata.get('target_pos')
        task_type = msg.metadata.get('task_type', 'move_to')
        self.current_task = {
            'type': task_type,
            'target_pos': target_pos,
        }
        # Compute path
        if target_pos:
            self.path = env.grid.shortest_path(self.position, target_pos) or []

    def _follow_task(self, env) -> dict:
        """Follow current task path."""
        if not self.path or len(self.path) <= 1:
            # Arrived at target or path broken
            action = self._task_action_at_target()
            self.current_task = None
            self.path = []
            return action

        # Move along path
        next_pos = self.path[1]
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
        """Generate a report message."""
        local = obs.get('local_grid', [])
        victims = 0
        fires = 0
        blocked = 0
        for row in local:
            for cell in row:
                if isinstance(cell, dict):
                    victims += cell.get('num_victims', 0)
                    if cell.get('hazard') == 'FIRE':
                        fires += 1
                    if cell.get('blocked', False):
                        blocked += 1
        return make_report(
            self.agent_id, self.position,
            {'num_victims_nearby': victims, 'fires_nearby': fires, 'blocked_nearby': blocked},
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

    def _autonomous_action(self, obs: dict, env) -> dict:
        # Prioritize unexplored areas
        unexplored = self._find_nearest_unexplored(env)
        if unexplored:
            return self._move_toward(unexplored, env)
        # Scan current area
        if not env.grid.cells.get(self.position, None) or \
           not env.grid.cells[self.position].explored:
            return {'type': 'scan'}
        return self._random_move(env)

    def _find_nearest_unexplored(self, env) -> Optional[Tuple[int, int]]:
        """BFS for nearest unexplored road cell."""
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
