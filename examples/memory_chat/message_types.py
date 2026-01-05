from dataclasses import dataclass
from enum import Enum
import time


class MessageRole(str, Enum):
    """Enum for message roles."""
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"
    TOOL = "tool"


@dataclass
class Message:
    """Data class for chat messages."""
    role: MessageRole
    content: str
    timestamp: float = time.time()
