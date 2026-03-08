"""
Message protocol: typed messages, message bus with configurable dropout.
Supports semantic (natural language) and raw (state codes) modes.
"""

import numpy as np
from enum import Enum
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any


class MessageType(Enum):
    REPORT = "report"
    TASK_ASSIGNMENT = "task_assignment"
    STATUS_UPDATE = "status_update"
    EMERGENCY = "emergency"
    ACKNOWLEDGEMENT = "ack"


@dataclass
class Message:
    sender: str
    receiver: str  # agent_id or "commander" or "broadcast"
    msg_type: MessageType
    content: str  # Semantic mode: natural language; Raw mode: encoded
    step: int = 0
    priority: int = 0  # Higher = more important
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_semantic(self) -> str:
        """Format as natural language string."""
        return f"[{self.msg_type.value}] From {self.sender}: {self.content}"

    def to_raw(self) -> str:
        """Format as compact code string."""
        return f"{self.msg_type.value}|{self.sender}|{self.content}"

    def to_dict(self) -> dict:
        return {
            'sender': self.sender,
            'receiver': self.receiver,
            'type': self.msg_type.value,
            'content': self.content,
            'step': self.step,
            'priority': self.priority,
        }


class MessageBus:
    """
    Central message queue with configurable dropout.
    Agents push messages; commander/agents pull each step.
    """

    def __init__(self, dropout_rate: float = 0.0,
                 mode: str = 'semantic',
                 max_msg_length: int = 200,
                 seed: int = 42):
        """
        Args:
            dropout_rate: Probability of dropping each message (0.0-1.0)
            mode: 'semantic' (natural language) or 'raw' (state codes)
            max_msg_length: Max characters per message content
        """
        self.dropout_rate = dropout_rate
        self.mode = mode
        self.max_msg_length = max_msg_length
        self.rng = np.random.default_rng(seed)
        self.queue: List[Message] = []
        self.delivered: List[Message] = []
        self.dropped: List[Message] = []
        self.total_sent = 0
        self.total_dropped = 0

    def send(self, message: Message):
        """Submit a message to the bus."""
        # Truncate content
        message.content = message.content[:self.max_msg_length]
        self.total_sent += 1

        # Apply dropout
        if self.rng.random() < self.dropout_rate:
            self.dropped.append(message)
            self.total_dropped += 1
            return False  # Message was dropped

        self.queue.append(message)
        return True  # Message delivered

    def receive(self, receiver_id: str) -> List[Message]:
        """Pull all messages addressed to receiver_id (or broadcast)."""
        messages = []
        remaining = []
        for msg in self.queue:
            if msg.receiver == receiver_id or msg.receiver == 'broadcast':
                messages.append(msg)
                self.delivered.append(msg)
            else:
                remaining.append(msg)
        self.queue = remaining

        # Format based on mode
        return messages

    def receive_all(self) -> List[Message]:
        """Pull all messages from the queue (for commander)."""
        messages = list(self.queue)
        self.delivered.extend(messages)
        self.queue = []
        return messages

    def format_messages(self, messages: List[Message]) -> str:
        """Format messages as a single string based on mode."""
        if self.mode == 'semantic':
            return '\n'.join(m.to_semantic() for m in messages)
        else:
            return '\n'.join(m.to_raw() for m in messages)

    def get_stats(self) -> dict:
        return {
            'total_sent': self.total_sent,
            'total_dropped': self.total_dropped,
            'total_delivered': len(self.delivered),
            'drop_rate_actual': self.total_dropped / max(1, self.total_sent),
            'queue_size': len(self.queue),
        }

    def clear(self):
        self.queue = []
        self.delivered = []
        self.dropped = []
        self.total_sent = 0
        self.total_dropped = 0


# ---- Template helpers for agents ----

def make_report(agent_id: str, position: tuple, observations: dict, step: int) -> Message:
    """Create a standard field agent report."""
    # Backward/forward compatible summary extraction.
    # Newer reports may provide a nested "summary" dict and rich metadata.
    summary = observations.get('summary', observations)
    victims = summary.get('num_victims_nearby', observations.get('num_victims_nearby', 0))
    fires = summary.get('fires_nearby', observations.get('fires_nearby', 0))
    blocked = summary.get('blocked_nearby', observations.get('blocked_nearby', 0))

    content = (f"Agent {agent_id} at ({position[0]},{position[1]}): "
               f"{victims} victims, {fires} fires, {blocked} blocked roads nearby.")

    return Message(
        sender=agent_id,
        receiver='commander',
        msg_type=MessageType.REPORT,
        content=content,
        step=step,
        # Preserve full metadata payload (position, observation, radius, findings, etc.)
        metadata=observations,
    )


def make_task_assignment(target_agent: str, task_type: str,
                         target_pos: tuple, step: int,
                         details: str = '',
                         path: Optional[List[tuple]] = None) -> Message:
    """Create a commander task assignment."""
    content = f"TASK: {task_type} at ({target_pos[0]},{target_pos[1]}). {details}"

    return Message(
        sender='commander',
        receiver=target_agent,
        msg_type=MessageType.TASK_ASSIGNMENT,
        content=content,
        step=step,
        priority=1,
        metadata={
            'task_type': task_type,
            'target_pos': target_pos,
            'path': path or [],
        },
    )


def make_emergency(agent_id: str, position: tuple, description: str, step: int) -> Message:
    """Create an emergency alert."""
    content = f"EMERGENCY from {agent_id} at ({position[0]},{position[1]}): {description}"
    return Message(
        sender=agent_id,
        receiver='commander',
        msg_type=MessageType.EMERGENCY,
        content=content,
        step=step,
        priority=2,
        metadata={'position': position},
    )
