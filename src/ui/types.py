from enum import Enum
from dataclasses import dataclass
import time


class MessageRole(str, Enum):
    """Enum for message roles."""
    USER = "user"
    ASSISTANT = "assistant"


@dataclass
class Message:
    """Data class for chat messages."""
    role: MessageRole
    content: str
    timestamp: float = time.time()