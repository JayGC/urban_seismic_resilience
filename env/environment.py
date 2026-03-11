"""
UrbanDisasterEnv: Gymnasium-style environment for the urban seismic disaster simulation.
Supports reset(config), step(actions), render().
"""

import numpy as np
import yaml
import copy
from typing import Dict, List, Tuple, Optional, Any

from .grid import Grid, CellType, HazardType, Victim, Building
from .seismic import SeismicModel


class UrbanDisasterEnv:
    """
    Main simulation environment.

    State: Grid of cells with buildings, roads, hazards, victims.
    Agents: Placed on road cells; take actions each step.
    Dynamics: Seismic damage, aftershocks, fire spread, victim health decay.
    """

    def __init__(self, config: Optional[dict] = None, config_path: Optional[str] = None):
        if config_path:
            with open(config_path) as f:
                config = yaml.safe_load(f)
        self.config = config or self._default_config()
        self.grid: Optional[Grid] = None
        self.seismic: Optional[SeismicModel] = None
        self.step_count = 0
        self.max_steps = self.config.get('max_steps', 50)
        self.done = False
        self.agent_positions: Dict[str, Tuple[int, int]] = {}
        self.event_log: List[dict] = []

    @staticmethod
    def _default_config() -> dict:
        return {
            'grid_width': 50,
            'grid_height': 50,
            'building_density': 0.25,
            'num_victims': 30,
            'num_fires': 10,
            'victim_decay_rate': 1.5,
            'fire_spread_prob': 0.05,
            'max_steps': 50,
            'seed': 42,
            'seismic': {
                'epicenter': [25, 25],
                'magnitude': 6.5,
                'decay_k': 0.05,
                'intensity_scale': 1.0,
                'aftershocks': [
                    {'step': 8, 'magnitude': 5.0},
                    {'step': 15, 'magnitude': 4.5},
                ],
                'black_swans': [],
            },
        }

    def reset(self, config: Optional[dict] = None) -> dict:
        """Reset the environment with a (possibly new) config. Returns initial observation."""
        if config:
            self.config = config
        seed = self.config.get('seed', 42)
        self.step_count = 0
        self.done = False
        self.event_log = []
        self.agent_positions = {}

        # Build grid
        self.grid = Grid(
            width=self.config['grid_width'],
            height=self.config['grid_height'],
            seed=seed,
        )
        self.grid.generate_city_layout(building_density=self.config.get('building_density', 0.25))

        # Seismic model
        self.seismic = SeismicModel(self.config.get('seismic', {}), seed=seed)

        # Apply initial earthquake damage
        damage_matrix = self.seismic.get_initial_damage(self.grid.width, self.grid.height)
        self._apply_damage_to_buildings(damage_matrix)
        self._process_collapses()

        # Place victims and fires
        self.grid.place_victims(self.config.get('num_victims', 30))
        self.grid.place_fires(self.config.get('num_fires', 10))

        self.event_log.append({
            'step': 0,
            'event': 'earthquake',
            'magnitude': self.seismic.magnitude,
            'epicenter': self.seismic.epicenter,
        })

        return self._get_global_obs()

    def step(self, actions: Dict[str, dict]) -> Tuple[dict, dict, bool, dict]:
        """
        Execute one simulation step.

        Args:
            actions: Dict of agent_id -> action_dict.
                     Action dict has 'type' and action-specific params.

        Returns:
            observation, rewards, done, info
        """
        if self.done:
            return self._get_global_obs(), {}, True, {'reason': 'already_done'}

        self.step_count += 1
        rewards = {}
        step_events = []

        # 1. Process agent actions
        for agent_id, action in actions.items():
            reward, events = self._execute_action(agent_id, action)
            rewards[agent_id] = reward
            step_events.extend(events)

        # 2. Aftershock check
        if self.seismic:
            aftershock_damage = self.seismic.get_aftershock_damage(
                self.step_count, self.grid.width, self.grid.height)
            if aftershock_damage is not None:
                self._apply_damage_to_buildings(aftershock_damage)
                new_collapses = self._process_collapses()
                step_events.append({
                    'type': 'aftershock',
                    'step': self.step_count,
                    'new_collapses': new_collapses,
                })

            # Black swan events
            for bs in self.seismic.get_black_swan_events(self.step_count):
                self._apply_black_swan(bs)
                step_events.append({'type': 'black_swan', 'step': self.step_count, 'details': bs})

        # 3. Fire spread
        self._spread_fires()
        self._process_fires()

        # 4. Victim health decay — only for people in hazardous buildings
        #    A person is a victim if ANY cell of their building has fire/debris/collapsed
        decay_rate = self.config.get('victim_decay_rate', 1.5)
        for pos, cell in self.grid.cells.items():
            if cell.victims and self.grid.is_cell_in_danger(pos):
                for v in cell.victims:
                    rate = decay_rate * (2.0 if cell.hazard == HazardType.FIRE else 1.0)
                    v.tick(rate)

        # 5. Check termination
        if self.step_count >= self.max_steps:
            self.done = True

        # Early termination: no living victims remain on the map
        metrics = self.get_metrics()
        if metrics['alive_unrescued'] == 0 and metrics['total_victims'] > 0:
            self.done = True
            print(f"\n** All victims accounted for at step {self.step_count}: "
                  f"{metrics['rescued']} rescued, {metrics['dead']} dead. Ending early. **\n")

        # Log
        self.event_log.append({
            'step': self.step_count,
            'events': step_events,
            'metrics': self.get_metrics(),
        })

        obs = self._get_global_obs()
        info = {'step': self.step_count, 'events': step_events}
        return obs, rewards, self.done, info

    def _execute_action(self, agent_id: str, action: dict) -> Tuple[float, List[dict]]:
        """Execute a single agent's action. Returns (reward, events)."""
        action_type = action.get('type', 'noop')
        reward = 0.0
        events = []

        pos = self.agent_positions.get(agent_id)
        if pos is None:
            return 0.0, []

        if action_type == 'move':
            dx, dy = action.get('dx', 0), action.get('dy', 0)
            new_pos = (pos[0] + dx, pos[1] + dy)
            if new_pos in self.grid.cells and self.grid.cells[new_pos].passable:
                self.agent_positions[agent_id] = new_pos
            else:
                reward -= 0.1  # Penalty for invalid move

        elif action_type == 'scan':
            # Mark nearby cells as explored
            x, y = pos
            for dx in range(-1, 2):
                for dy in range(-1, 2):
                    p = (x + dx, y + dy)
                    if p in self.grid.cells:
                        self.grid.cells[p].explored = True
            events.append({'type': 'scan', 'agent': agent_id, 'pos': pos})

        elif action_type == 'rescue':
            # Rescue victims in adjacent cells (radius 1 = 3x3, matches observation radius)
            # Only rescue people who are in danger (building has fire/debris/collapsed)
            x, y = pos
            rescued_count = 0
            for dx in range(-1, 2):
                for dy in range(-1, 2):
                    p = (x + dx, y + dy)
                    if p in self.grid.cells and self.grid.is_cell_in_danger(p):
                        cell = self.grid.cells[p]
                        for v in cell.victims:
                            if not v.rescued and v.health > 0:
                                v.rescued = True
                                v.trapped = False
                                rescued_count += 1
            reward += rescued_count * 10.0
            if rescued_count > 0:
                events.append({'type': 'rescue', 'agent': agent_id, 'count': rescued_count})

        elif action_type == 'extinguish':
            # Reduce fire in adjacent cells
            x, y = pos
            for dx in range(-1, 2):
                for dy in range(-1, 2):
                    p = (x + dx, y + dy)
                    if p in self.grid.cells:
                        cell = self.grid.cells[p]
                        if cell.hazard == HazardType.FIRE:
                            cell.fire_intensity = max(0, cell.fire_intensity - 25)
                            if cell.fire_intensity <= 0:
                                cell.hazard = HazardType.NONE
                                reward += 3.0
                                events.append({'type': 'fire_out', 'pos': p})

        elif action_type == 'treat':
            # Stabilize victims (slow health decay)
            x, y = pos
            for dx in range(-1, 2):
                for dy in range(-1, 2):
                    p = (x + dx, y + dy)
                    if p in self.grid.cells:
                        for v in self.grid.cells[p].victims:
                            if not v.rescued and v.health > 0:
                                v.health = min(100, v.health + 10)
                                reward += 1.0

        # 'noop' or 'send_report' -> no env effect
        return reward, events

    def _apply_damage_to_buildings(self, damage_matrix: np.ndarray):
        """Apply damage from a seismic event to all buildings."""
        for bid, building in self.grid.buildings.items():
            max_damage = 0
            for cx, cy in building.cells:
                if 0 <= cy < damage_matrix.shape[0] and 0 <= cx < damage_matrix.shape[1]:
                    max_damage = max(max_damage, damage_matrix[cy, cx])
            building.apply_damage(max_damage)

    def _process_collapses(self) -> int:
        """Check all buildings for collapse; apply spillover. Returns number of new collapses."""
        new_collapses = 0
        for bid, building in self.grid.buildings.items():
            if not building.collapsed and building.check_collapse(self.grid.rng):
                new_collapses += 1
                # Mark building cells
                for cx, cy in building.cells:
                    cell = self.grid.cells.get((cx, cy))
                    if cell:
                        cell.hazard = HazardType.DEBRIS
                        cell.blocked = True
                # Spillover to adjacent roads
                self.grid.apply_spillover(building)
        return new_collapses

    def _spread_fires(self):
        """Probabilistic fire spread to adjacent cells."""
        spread_prob = self.config.get('fire_spread_prob', 0.05)
        new_fires = []
        for (x, y), cell in self.grid.cells.items():
            if cell.hazard == HazardType.FIRE and cell.fire_intensity > 20:
                for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                    nb = (x + dx, y + dy)
                    if nb in self.grid.cells:
                        nc = self.grid.cells[nb]
                        if nc.cell_type == CellType.BUILDING and nc.hazard == HazardType.NONE:
                            if self.grid.rng.random() < spread_prob:
                                new_fires.append((nb, cell.fire_intensity * 0.5))
        for pos, intensity in new_fires:
            self.grid.cells[pos].hazard = HazardType.FIRE
            self.grid.cells[pos].fire_intensity = intensity

    def _process_fires(self):
        "Mark all the buildings with any cell as fire to be on fire."
        for bid, building in self.grid.buildings.items:
            fire = False
            for cx, cy in building.cells:
                cell = self.grid.cells.get((cx, cy))
                if cell.hazard == HazardType.FIRE:
                    fire = True
                    break
            if fire:
                building.fire = True

    def _apply_black_swan(self, event: dict):
        """Apply a black swan event (sudden collapse, fire, etc.)."""
        etype = event.get('type', 'collapse')
        if etype == 'collapse':
            bid = event.get('building_id')
            if bid in self.grid.buildings:
                b = self.grid.buildings[bid]
                b.integrity = 0
                b.collapsed = True
                for cx, cy in b.cells:
                    cell = self.grid.cells.get((cx, cy))
                    if cell:
                        cell.hazard = HazardType.DEBRIS
                        cell.blocked = True
                self.grid.apply_spillover(b)
        elif etype == 'fire':
            pos = tuple(event.get('position', [0, 0]))
            if pos in self.grid.cells:
                self.grid.cells[pos].hazard = HazardType.FIRE
                self.grid.cells[pos].fire_intensity = event.get('intensity', 70)

    # ---- Observations ----

    def _get_global_obs(self) -> dict:
        """Full global observation (for logging / analysis)."""
        return {
            'step': self.step_count,
            'agent_positions': dict(self.agent_positions),
            'metrics': self.get_metrics(),
        }

    def get_agent_observation(self, agent_id: str, radius: int = 1) -> dict:
        """Local observation for a field agent."""
        pos = self.agent_positions.get(agent_id, (0, 0))
        local_grid = self.grid.get_local_observation(pos[0], pos[1], radius)
        return {
            'position': pos,
            'local_grid': local_grid,
            'agent_id': agent_id,
            'step': self.step_count,
        }

    def get_commander_observation(self, zone_size: int = 10) -> dict:
        """Coarse zone summaries for the commander agent."""
        zones = []
        for zy in range(0, self.grid.height, zone_size):
            for zx in range(0, self.grid.width, zone_size):
                zones.append(self.grid.get_zone_summary(zx, zy, zone_size))
        return {
            'step': self.step_count,
            'zones': zones,
            'agent_positions': dict(self.agent_positions),
            'num_agents': len(self.agent_positions),
        }

    # ---- Metrics ----

    def get_metrics(self) -> dict:
        """Compute current metrics."""
        total_people = 0
        total_victims = 0
        rescued = 0
        alive_trapped = 0
        dead = 0
        safe_people = 0

        for pos, cell in self.grid.cells.items():
            for v in cell.victims:
                total_people += 1
                if self.grid.is_cell_in_danger(pos):
                    # This person is a victim (in a hazardous building)
                    total_victims += 1
                    if v.rescued:
                        rescued += 1
                    elif v.health > 0:
                        alive_trapped += 1
                    else:
                        dead += 1
                else:
                    safe_people += 1

        fires = sum(1 for c in self.grid.cells.values() if c.hazard == HazardType.FIRE)
        blocked = sum(1 for c in self.grid.cells.values()
                     if c.cell_type == CellType.ROAD and c.blocked)
        collapsed = sum(1 for b in self.grid.buildings.values() if b.collapsed)

        return {
            'step': self.step_count,
            'total_people': total_people,
            'safe_people': safe_people,
            'total_victims': total_victims,
            'rescued': rescued,
            'alive_unrescued': alive_trapped,
            'dead': dead,
            'survival_rate': rescued / max(1, total_victims),
            'active_fires': fires,
            'blocked_roads': blocked,
            'collapsed_buildings': collapsed,
        }

    # ---- Agent placement ----

    def place_agents(self, agent_configs: List[dict]):
        """
        Place agents on the grid.
        agent_configs: list of {'id': str, 'type': str, 'position': (x,y) or 'random'}
        """
        road_cells = [(x, y) for (x, y), c in self.grid.cells.items() if c.passable]
        for ac in agent_configs:
            aid = ac['id']
            pos = ac.get('position', 'random')
            if pos == 'random' or pos is None:
                idx = self.grid.rng.integers(0, len(road_cells))
                pos = road_cells[idx]
            self.agent_positions[aid] = tuple(pos)

    # ---- Rendering ----

    def render(self, show: bool = True, save_path: Optional[str] = None,
               figsize: Tuple[int, int] = (12, 12)):
        """Render the current grid state as a 2D matplotlib figure."""
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
        from matplotlib.colors import ListedColormap

        w, h = self.grid.width, self.grid.height
        img = np.zeros((h, w, 3))

        # Base colors
        for (x, y), cell in self.grid.cells.items():
            if cell.cell_type == CellType.BUILDING:
                if cell.building_id is not None and self.grid.buildings[cell.building_id].collapsed:
                    img[y, x] = [0.3, 0.3, 0.3]  # Collapsed = dark grey
                else:
                    img[y, x] = [0.6, 0.6, 0.7]  # Building = blue-grey
            elif cell.blocked:
                img[y, x] = [0.5, 0.4, 0.3]  # Blocked road = brown
            else:
                img[y, x] = [0.9, 0.9, 0.85]  # Road = light

            # Hazard overlays
            if cell.hazard == HazardType.FIRE:
                intensity = cell.fire_intensity / 100.0
                img[y, x] = [1.0, 0.3 * (1 - intensity), 0.0]
            elif cell.hazard == HazardType.DEBRIS:
                img[y, x] = [0.4, 0.35, 0.3]

        fig, ax = plt.subplots(1, 1, figsize=figsize)
        ax.imshow(img, origin='lower', interpolation='nearest')

        # --- Draw people/victims as markers with health timer rings ---
        from matplotlib.patches import Arc
        for (x, y), cell in self.grid.cells.items():
            if not cell.victims:
                continue
            is_hazardous = self.grid.is_cell_in_danger((x, y))
            alive_unrescued = [v for v in cell.victims if not v.rescued and v.health > 0]
            rescued = [v for v in cell.victims if v.rescued]
            dead = [v for v in cell.victims if not v.rescued and v.health <= 0]

            if alive_unrescued:
                n = len(alive_unrescued)
                if is_hazardous:
                    # Victim in danger — yellow diamond with health timer ring
                    ax.plot(x, y, 'D', color='#facc15', markersize=9,
                            markeredgecolor='black', markeredgewidth=0.8)
                    # Health timer ring: arc proportional to avg health
                    avg_health = sum(v.health for v in alive_unrescued) / n
                    sweep = 360 * (avg_health / 100.0)
                    ring_color = '#22c55e' if avg_health > 50 else '#eab308' if avg_health > 25 else '#ef4444'
                    arc = Arc((x, y), 1.4, 1.4, angle=90, theta1=0, theta2=sweep,
                              color=ring_color, linewidth=2.5)
                    ax.add_patch(arc)
                    # Always show victim count inside the marker
                    ax.text(x, y, str(n), ha='center', va='center',
                            fontsize=6, color='black', fontweight='bold')
                else:
                    # Safe person — white circle, no timer
                    ax.plot(x, y, 'o', color='white', markersize=6,
                            markeredgecolor='#64748b', markeredgewidth=0.8)
                    if n > 1:
                        ax.text(x, y, str(n), ha='center', va='center',
                                fontsize=5, color='#334155', fontweight='bold')
            elif rescued:
                # Rescued — green-teal diamond with count
                n = len(rescued)
                ax.plot(x, y, 'D', color='#00ff80', markersize=8,
                        markeredgecolor='black', markeredgewidth=0.8)
                ax.text(x, y, str(n), ha='center', va='center',
                        fontsize=6, color='black', fontweight='bold')
            elif dead:
                # Dead — grey X with count
                n = len(dead)
                ax.plot(x, y, 'X', color='#6b7280', markersize=7,
                        markeredgecolor='#374151', markeredgewidth=0.8)
                if n > 1:
                    ax.text(x, y + 0.55, str(n), ha='center', va='bottom',
                            fontsize=5, color='#9ca3af', fontweight='bold')

        # Agent markers
        colors = {'scout': 'cyan', 'firefighter': 'red', 'medic': 'green', 'commander': 'white'}
        for agent_id, pos in self.agent_positions.items():
            atype = agent_id.split('_')[0] if '_' in agent_id else 'scout'
            color = colors.get(atype, 'cyan')
            ax.plot(pos[0], pos[1], 'o', color=color, markersize=8,
                   markeredgecolor='black', markeredgewidth=1)

        ax.set_title(f'Step {self.step_count} | {self.get_metrics()["rescued"]} rescued '
                     f'| {self.get_metrics()["active_fires"]} fires '
                     f'| {self.get_metrics()["collapsed_buildings"]} collapsed')

        # Legend
        from matplotlib.lines import Line2D
        legend_items = [
            mpatches.Patch(color=[0.9, 0.9, 0.85], label='Road'),
            mpatches.Patch(color=[0.6, 0.6, 0.7], label='Building'),
            mpatches.Patch(color=[0.3, 0.3, 0.3], label='Collapsed'),
            mpatches.Patch(color=[0.5, 0.4, 0.3], label='Blocked'),
            mpatches.Patch(color=[1.0, 0.3, 0.0], label='Fire'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='white',
                   markeredgecolor='#64748b', markersize=8, linestyle='None', label='Person (Safe)'),
            Line2D([0], [0], marker='D', color='w', markerfacecolor='#facc15',
                   markeredgecolor='black', markersize=8, linestyle='None', label='Victim (Alive)'),
            Line2D([0], [0], marker='D', color='w', markerfacecolor='#00ff80',
                   markeredgecolor='black', markersize=8, linestyle='None', label='Victim (Rescued)'),
            Line2D([0], [0], marker='X', color='w', markerfacecolor='#6b7280',
                   markeredgecolor='#374151', markersize=8, linestyle='None', label='Victim (Dead)'),
        ]
        ax.legend(handles=legend_items, loc='upper right', fontsize=8)

        ax.set_xlim(-0.5, w - 0.5)
        ax.set_ylim(-0.5, h - 0.5)

        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches='tight')
        if show:
            plt.show()
        else:
            plt.close(fig)
        return fig

    def render_mental_map(self, mental_map, show: bool = True, save_path: Optional[str] = None,
                         figsize: Tuple[int, int] = (12, 12)):
        """Render the commander's mental map (beliefs about the grid)."""
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
        from matplotlib.colors import ListedColormap

        w, h = mental_map.width, mental_map.height
        img = np.zeros((h, w, 3))

        # Base colors - showing structural layout known from city maps
        # Dynamic info (blockages, collapses) only update when observed
        for (x, y), cell in mental_map.cells.items():
            # Show structural information (always visible - known from city maps)
            if cell.cell_type == CellType.BUILDING:
                # Buildings are gray-blue, but dark gray if observed to be collapsed
                if cell.explored and cell.building_id is not None and mental_map.buildings[cell.building_id].collapsed:
                    img[y, x] = [0.3, 0.3, 0.3]  # Collapsed building (observed)
                else:
                    img[y, x] = [0.6, 0.6, 0.7]  # Building (default/not yet observed collapse status)
            elif cell.cell_type == CellType.ROAD:
                # Roads are light, but brown if observed to be blocked
                if cell.explored and cell.blocked is True:
                    img[y, x] = [0.5, 0.4, 0.3]  # Blocked road (observed)
                else:
                    img[y, x] = [0.9, 0.9, 0.85]  # Road (default/not yet observed blockage)
            else:
                # Unknown structure type (shouldn't happen after init)
                img[y, x] = [0.1, 0.1, 0.1]

            # Hazard overlays (only show if explored)
            if cell.explored:
                if cell.hazard == HazardType.FIRE:
                    intensity = cell.fire_intensity / 100.0 if cell.fire_intensity else 0
                    img[y, x] = [1.0, 0.3 * (1 - intensity), 0.0]  # Fire (observed)
                elif cell.hazard == HazardType.DEBRIS:
                    img[y, x] = [0.4, 0.35, 0.3]

        fig, ax = plt.subplots(1, 1, figsize=figsize)
        ax.imshow(img, origin='lower', interpolation='nearest')

        # --- Draw known victims as markers with health timer rings ---
        from matplotlib.patches import Arc
        for (x, y), cell in mental_map.cells.items():
            if not cell.explored or not cell.victims:
                continue
            is_hazardous = cell.hazard in (HazardType.FIRE, HazardType.DEBRIS)
            alive_unrescued = [v for v in cell.victims if not v.rescued and v.health > 0]
            rescued = [v for v in cell.victims if v.rescued]
            dead = [v for v in cell.victims if not v.rescued and v.health <= 0]

            if alive_unrescued:
                n = len(alive_unrescued)
                if is_hazardous:
                    ax.plot(x, y, 'D', color='#facc15', markersize=9,
                            markeredgecolor='black', markeredgewidth=0.8)
                    avg_health = sum(v.health for v in alive_unrescued) / n
                    sweep = 360 * (avg_health / 100.0)
                    ring_color = '#22c55e' if avg_health > 50 else '#eab308' if avg_health > 25 else '#ef4444'
                    arc = Arc((x, y), 1.4, 1.4, angle=90, theta1=0, theta2=sweep,
                              color=ring_color, linewidth=2.5)
                    ax.add_patch(arc)
                    ax.text(x, y, str(n), ha='center', va='center',
                            fontsize=6, color='black', fontweight='bold')
                else:
                    ax.plot(x, y, 'o', color='white', markersize=6,
                            markeredgecolor='#64748b', markeredgewidth=0.8)
                    if n > 1:
                        ax.text(x, y, str(n), ha='center', va='center',
                                fontsize=5, color='#334155', fontweight='bold')
            elif rescued:
                n = len(rescued)
                ax.plot(x, y, 'D', color='#00ff80', markersize=8,
                        markeredgecolor='black', markeredgewidth=0.8)
                ax.text(x, y, str(n), ha='center', va='center',
                        fontsize=6, color='black', fontweight='bold')
            elif dead:
                n = len(dead)
                ax.plot(x, y, 'X', color='#6b7280', markersize=7,
                        markeredgecolor='#374151', markeredgewidth=0.8)
                if n > 1:
                    ax.text(x, y + 0.55, str(n), ha='center', va='bottom',
                            fontsize=5, color='#9ca3af', fontweight='bold')

        # Agent markers (if available)
        if hasattr(self, 'agent_positions'):
            colors = {'scout': 'cyan', 'firefighter': 'red', 'medic': 'green', 'commander': 'white'}
            for agent_id, pos in self.agent_positions.items():
                atype = agent_id.split('_')[0] if '_' in agent_id else 'scout'
                color = colors.get(atype, 'cyan')
                ax.plot(pos[0], pos[1], 'o', color=color, markersize=8,
                       markeredgecolor='black', markeredgewidth=1)

        # Get stats for title
        exploration_pct = mental_map.get_explored_fraction() * 100
        known_victims = len(mental_map.get_all_known_victims())

        ax.set_title(f'Commander\'s Mental Map | Step {mental_map.current_step} | '
                     f'Exploration: {exploration_pct:.1f}% | Known Victims: {known_victims}')

        # Legend
        from matplotlib.lines import Line2D
        legend_items = [
            mpatches.Patch(color=[0.9, 0.9, 0.85], label='Road (Known from maps)'),
            mpatches.Patch(color=[0.5, 0.4, 0.3], label='Road Blocked (Observed)'),
            mpatches.Patch(color=[0.6, 0.6, 0.7], label='Building (Known from maps)'),
            mpatches.Patch(color=[0.3, 0.3, 0.3], label='Building Collapsed (Observed)'),
            mpatches.Patch(color=[1.0, 0.3, 0.0], label='Fire (Observed)'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='white',
                   markeredgecolor='#64748b', markersize=8, linestyle='None', label='Person (Safe)'),
            Line2D([0], [0], marker='D', color='w', markerfacecolor='#facc15',
                   markeredgecolor='black', markersize=8, linestyle='None', label='Victim (Alive)'),
            Line2D([0], [0], marker='D', color='w', markerfacecolor='#00ff80',
                   markeredgecolor='black', markersize=8, linestyle='None', label='Victim (Rescued)'),
            Line2D([0], [0], marker='X', color='w', markerfacecolor='#6b7280',
                   markeredgecolor='#374151', markersize=8, linestyle='None', label='Victim (Dead)'),
        ]
        ax.legend(handles=legend_items, loc='upper right', fontsize=8)

        ax.set_xlim(-0.5, w - 0.5)
        ax.set_ylim(-0.5, h - 0.5)

        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches='tight')
        if show:
            plt.show()
        else:
            plt.close(fig)
        return fig
