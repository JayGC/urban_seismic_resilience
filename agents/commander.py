"""
Commander agents: coordinate field agents by issuing task assignments.
- HeuristicCommander: rule-based prioritization
- LLMCommander: uses GPT/Claude API for strategic planning
"""

import json
import re
from typing import Dict, List, Tuple, Optional, Any
from .messages import Message, MessageBus, MessageType, make_task_assignment


class CommanderAgent:
    """Base commander interface."""

    def __init__(self, agent_ids: List[str]):
        self.agent_ids = agent_ids
        self.mental_map: Dict[str, dict] = {}  # agent_id -> last known info
        self.zone_data: List[dict] = []
        self.step = 0
        self.assignments: Dict[str, dict] = {}  # agent_id -> current assignment
        self.replan_log: List[dict] = []

    def update_mental_map(self, messages: List[Message]):
        """Update internal model from agent reports."""
        for msg in messages:
            if msg.msg_type in (MessageType.REPORT, MessageType.EMERGENCY):
                self.mental_map[msg.sender] = {
                    'content': msg.content,
                    'metadata': msg.metadata,
                    'step': msg.step,
                }

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
        self.zone_data = observation.get('zones', [])
        self.update_mental_map(messages)

        commands = []

        # Categorize agents
        scouts = [a for a in self.agent_ids if 'scout' in a.lower()]
        firefighters = [a for a in self.agent_ids if 'firefighter' in a.lower()]
        medics = [a for a in self.agent_ids if 'medic' in a.lower()]

        # Sort zones by priority
        fire_zones = sorted(self.zone_data, key=lambda z: z.get('fires', 0), reverse=True)
        victim_zones = sorted(self.zone_data, key=lambda z: z.get('victims_alive', 0), reverse=True)
        unexplored_zones = self.zone_data  # All zones are candidates for exploration

        # Assign firefighters to fire zones
        for i, agent_id in enumerate(firefighters):
            if i < len(fire_zones) and fire_zones[i].get('fires', 0) > 0:
                zone = fire_zones[i]
                target = (zone['zone'][0] + 5, zone['zone'][1] + 5)  # Zone center
                cmd = make_task_assignment(
                    agent_id, 'extinguish_fire', target, self.step,
                    f"Zone has {zone['fires']} active fires."
                )
                commands.append(cmd)
                self.assignments[agent_id] = {'task': 'extinguish', 'zone': zone['zone']}

        # Assign medics to victim zones
        for i, agent_id in enumerate(medics):
            if i < len(victim_zones) and victim_zones[i].get('victims_alive', 0) > 0:
                zone = victim_zones[i]
                target = (zone['zone'][0] + 5, zone['zone'][1] + 5)
                cmd = make_task_assignment(
                    agent_id, 'rescue_victims', target, self.step,
                    f"Zone has {zone['victims_alive']} victims alive."
                )
                commands.append(cmd)
                self.assignments[agent_id] = {'task': 'rescue', 'zone': zone['zone']}

        # Assign scouts to explore
        for i, agent_id in enumerate(scouts):
            if i < len(unexplored_zones):
                zone = unexplored_zones[i]
                target = (zone['zone'][0] + 5, zone['zone'][1] + 5)
                cmd = make_task_assignment(
                    agent_id, 'search_zone', target, self.step,
                    f"Explore this zone."
                )
                commands.append(cmd)
                self.assignments[agent_id] = {'task': 'scout', 'zone': zone['zone']}

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
        self.api_key = api_key
        self.model = model
        self.provider = provider
        self.fallback = HeuristicCommander(agent_ids)
        self.llm_call_count = 0
        self.total_tokens = 0

    def _build_prompt(self, observation: dict, messages: List[Message]) -> str:
        """Build the LLM prompt from observation and messages."""
        zones_summary = ""
        for z in observation.get('zones', []):
            zones_summary += (
                f"  Zone ({z['zone'][0]},{z['zone'][1]}): "
                f"victims_alive={z['victims_alive']}, dead={z['victims_dead']}, "
                f"rescued={z['victims_rescued']}, fires={z['fires']}, "
                f"blocked={z['blocked_roads']}, collapsed={z['collapsed_buildings']}\n"
            )

        agent_reports = ""
        for msg in messages:
            agent_reports += f"  {msg.to_semantic()}\n"

        agent_positions = observation.get('agent_positions', {})
        agents_info = ""
        for aid, pos in agent_positions.items():
            atype = self._get_agent_type(aid)
            agents_info += f"  {aid} ({atype}) at ({pos[0]},{pos[1]})\n"

        prompt = f"""You are the Commander of an urban disaster response operation after an earthquake.
Your role is to assign tasks to field agents to maximize civilian survival.

CURRENT STEP: {observation.get('step', 0)}

ZONE SUMMARIES:
{zones_summary}

AGENT POSITIONS:
{agents_info}

AGENT REPORTS:
{agent_reports if agent_reports else "  No reports this step."}

AVAILABLE AGENTS: {', '.join(self.agent_ids)}

AGENT TYPES AND CAPABILITIES:
- scout_*: Can scan/explore areas, fast movement
- firefighter_*: Can extinguish fires
- medic_*: Can rescue and treat victims

Respond with a JSON array of task assignments. Each assignment:
{{
  "agent_id": "<agent_id>",
  "task_type": "search_zone|rescue_victims|extinguish_fire|move_to",
  "target_x": <int>,
  "target_y": <int>,
  "reason": "<brief reason>"
}}

PRIORITIES:
1. Save living victims (assign medics to zones with most alive victims)
2. Extinguish fires near victims (assign firefighters)
3. Explore unknown areas (assign scouts)
4. Consider blocked roads and plan routes around them

Respond ONLY with the JSON array, no other text."""

        return prompt

    def decide(self, observation: dict, messages: List[Message],
               env=None) -> List[Message]:
        self.step = observation.get('step', 0)
        self.zone_data = observation.get('zones', [])
        self.update_mental_map(messages)

        prompt = self._build_prompt(observation, messages)

        try:
            assignments = self._call_llm(prompt)
            commands = self._parse_assignments(assignments)
            if commands:
                return commands
        except Exception as e:
            pass  # Fall through to heuristic

        # Fallback
        self.fallback.agent_ids = self.agent_ids
        return self.fallback.decide(observation, messages, env)

    def _call_llm(self, prompt: str) -> str:
        """Call the LLM API. Override for different providers."""
        self.llm_call_count += 1

        if self.provider == 'openai' and self.api_key:
            import openai
            client = openai.OpenAI(api_key=self.api_key)
            response = client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a disaster response commander. Respond only with JSON."},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=500,
                temperature=0.3,
            )
            self.total_tokens += response.usage.total_tokens if response.usage else 0
            return response.choices[0].message.content

        elif self.provider == 'anthropic' and self.api_key:
            import anthropic
            client = anthropic.Anthropic(api_key=self.api_key)
            response = client.messages.create(
                model=self.model,
                max_tokens=500,
                messages=[{"role": "user", "content": prompt}],
                system="You are a disaster response commander. Respond only with JSON.",
            )
            return response.content[0].text

        else:
            # Simulate LLM response for testing (uses heuristic logic)
            return self._simulated_llm_response()

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
                assignments.append({
                    'agent_id': aid,
                    'task_type': 'search_zone',
                    'target_x': z['zone'][0] + 5,
                    'target_y': z['zone'][1] + 5,
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

        commands = []
        for a in assignments:
            agent_id = a.get('agent_id', '')
            if agent_id not in self.agent_ids:
                continue
            task_type = a.get('task_type', 'move_to')
            tx = a.get('target_x', 25)
            ty = a.get('target_y', 25)
            reason = a.get('reason', '')

            cmd = make_task_assignment(
                agent_id, task_type, (tx, ty), self.step, reason
            )
            commands.append(cmd)
            self.assignments[agent_id] = {
                'task': task_type, 'target': (tx, ty), 'reason': reason
            }

        return commands

    def get_stats(self) -> dict:
        return {
            'llm_calls': self.llm_call_count,
            'total_tokens': self.total_tokens,
            'current_assignments': dict(self.assignments),
        }
