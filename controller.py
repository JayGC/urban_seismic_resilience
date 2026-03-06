"""
Controller: orchestrates the simulation loop, connecting env, agents, and message bus.
Supports hierarchical (commander + field) and decentralized modes.
"""

import json
import os
from typing import Dict, List, Optional, Tuple, Any

from env.environment import UrbanDisasterEnv
from agents.messages import MessageBus, Message
from agents.field_agents import FieldAgent, ScoutAgent, FirefighterAgent, MedicAgent
from agents.commander import CommanderAgent, HeuristicCommander, LLMCommander


class SimulationController:
    """
    Runs the full simulation loop:
    1. Env step → observations
    2. Commander receives reports, issues commands
    3. Field agents observe + receive commands → produce actions
    4. Actions fed back to env
    """

    def __init__(self, config: dict):
        self.config = config
        self.env = UrbanDisasterEnv(config=config)
        self.message_bus = MessageBus(
            dropout_rate=config.get('dropout_rate', 0.0),
            mode=config.get('message_mode', 'semantic'),
            seed=config.get('seed', 42),
        )
        self.field_agents: Dict[str, FieldAgent] = {}
        self.commander: Optional[CommanderAgent] = None
        self.mode = config.get('controller_mode', 'hierarchical')  # or 'decentralized'
        self.trajectory: List[dict] = []
        self.step_count = 0

    def setup(self):
        """Initialize environment and agents."""
        self.env.reset()
        self._create_agents()

    def _create_agents(self):
        """Create field agents and commander based on config."""
        agent_configs = self.config.get('agents', {})
        num_scouts = agent_configs.get('num_scouts', 3)
        num_firefighters = agent_configs.get('num_firefighters', 3)
        num_medics = agent_configs.get('num_medics', 4)

        placement = []
        agent_ids = []

        for i in range(num_scouts):
            aid = f'scout_{i}'
            agent_ids.append(aid)
            placement.append({'id': aid, 'type': 'scout', 'position': 'random'})

        for i in range(num_firefighters):
            aid = f'firefighter_{i}'
            agent_ids.append(aid)
            placement.append({'id': aid, 'type': 'firefighter', 'position': 'random'})

        for i in range(num_medics):
            aid = f'medic_{i}'
            agent_ids.append(aid)
            placement.append({'id': aid, 'type': 'medic', 'position': 'random'})

        self.env.place_agents(placement)

        # Create field agent objects
        for aid in agent_ids:
            pos = self.env.agent_positions[aid]
            if 'scout' in aid:
                self.field_agents[aid] = ScoutAgent(aid, pos)
            elif 'firefighter' in aid:
                self.field_agents[aid] = FirefighterAgent(aid, pos)
            elif 'medic' in aid:
                self.field_agents[aid] = MedicAgent(aid, pos)

        # Create commander
        if self.mode == 'hierarchical':
            commander_type = self.config.get('commander_type', 'heuristic')
            if commander_type == 'llm':
                self.commander = LLMCommander(
                    agent_ids,
                    api_key=self.config.get('api_key'),
                    model=self.config.get('llm_model', 'api-gpt-oss-120b'),
                    provider=self.config.get('llm_provider', 'openai'),
                )
            else:
                self.commander = HeuristicCommander(agent_ids)

    def run_step(self) -> dict:
        """Execute one full step of the simulation."""
        self.step_count += 1

        # 1. Commander phase (if hierarchical)
        commander_commands = []
        if self.mode == 'hierarchical' and self.commander:
            # Commander receives reports from bus (no ground-truth env observation)
            reports = self.message_bus.receive_all()
            # Commander decides using ONLY mental map built from reports
            cmd_obs = {'step': self.env.step_count}  # No zones/agent_positions from env
            commander_commands = self.commander.decide(cmd_obs, reports, self.env)
            # Send commands through bus
            for cmd in commander_commands:
                self.message_bus.send(cmd)

        # 2. Field agent phase
        actions = {}
        all_outgoing = []

        for aid, agent in self.field_agents.items():
            # Agent observes
            obs = agent.observe(self.env)
            # Agent receives messages
            incoming = self.message_bus.receive(aid)
            # Agent decides
            action, outgoing = agent.decide(obs, incoming, self.env)
            actions[aid] = action
            all_outgoing.extend(outgoing)

        # Send field agent messages through bus
        for msg in all_outgoing:
            self.message_bus.send(msg)

        # 3. Environment step
        obs, rewards, done, info = self.env.step(actions)

        # 4. Log trajectory
        step_data = {
            'step': self.step_count,
            'actions': {aid: a.get('type', 'noop') for aid, a in actions.items()},
            'agent_positions': dict(self.env.agent_positions),
            'metrics': self.env.get_metrics(),
            'rewards': rewards,
            'events': info.get('events', []),
            'messages_sent': self.message_bus.total_sent,
            'messages_dropped': self.message_bus.total_dropped,
            'done': done,
        }
        self.trajectory.append(step_data)

        return step_data

    def run(self, max_steps: Optional[int] = None, verbose: bool = True) -> List[dict]:
        """Run the full simulation."""
        max_steps = max_steps or self.config.get('max_steps', 50)

        for step in range(max_steps):
            step_data = self.run_step()

            if verbose and step % 5 == 0:
                m = step_data['metrics']
                print(f"Step {step_data['step']:3d} | "
                      f"Rescued: {m['rescued']}/{m['total_victims']} | "
                      f"Alive: {m['alive_unrescued']} | Dead: {m['dead']} | "
                      f"Fires: {m['active_fires']} | "
                      f"Collapsed: {m['collapsed_buildings']}")

            if step_data['done']:
                break

        if verbose:
            final = self.trajectory[-1]['metrics']
            print(f"\n--- FINAL RESULTS ---")
            print(f"Survival rate: {final['survival_rate']:.2%}")
            print(f"Rescued: {final['rescued']}/{final['total_victims']}")
            print(f"Dead: {final['dead']}")
            print(f"Message stats: {self.message_bus.get_stats()}")

        return self.trajectory

    def save_trajectory(self, path: str):
        """Save trajectory to JSON."""
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)
        with open(path, 'w') as f:
            json.dump(self.trajectory, f, indent=2, default=str)

    def get_final_metrics(self) -> dict:
        """Return final step metrics."""
        if self.trajectory:
            return self.trajectory[-1]['metrics']
        return {}

    def get_agent_stats(self) -> dict:
        """Get per-agent statistics."""
        stats = {}
        for aid, agent in self.field_agents.items():
            stats[aid] = {
                'type': agent.agent_type,
                'idle_steps': agent.idle_steps,
                'total_steps': agent.total_steps,
                'idle_rate': agent.idle_steps / max(1, agent.total_steps),
                'actions': agent.actions_taken,
            }
        if self.commander and hasattr(self.commander, 'get_stats'):
            stats['commander'] = self.commander.get_stats()
        return stats
