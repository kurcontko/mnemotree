from abc import ABC, abstractmethod
import textwrap
from typing import List, Union, Protocol

from langchain.schema import Document

from ..core.models import MemoryItem


class MemoryLike(Protocol):
    """Protocol defining the interface for memory-like objects"""
    @property
    def content(self) -> str: ...

    @property
    def timestamp(self) -> str: ...

    @property
    def memory_type(self): ...

    @property
    def tags(self) -> List[str]: ...

class FormattingStrategy(ABC):
    """Abstract base class for different memory formatting strategies"""
    @abstractmethod
    def get_content(self, item: Union[Document, MemoryItem]) -> str:
        """Get the main content from the memory item"""
        pass

    @abstractmethod
    def get_metadata(self, item: Union[Document, MemoryItem]) -> tuple[str, str, List[str]]:
        """Get metadata (timestamp, memory_type, tags) from the memory item"""
        pass

class LangchainDocumentStrategy(FormattingStrategy):
    def get_content(self, doc: Document) -> str:
        return doc.page_content

    def get_metadata(self, doc: Document) -> tuple[str, str, List[str]]:
        return (
            doc.metadata.get('timestamp', 'N/A'),
            doc.metadata.get('memory_type', 'general'),
            doc.metadata.get('tags', [])
        )

class MemoryItemStrategy(FormattingStrategy):
    def get_content(self, memory: MemoryItem) -> str:
        return memory.content

    def get_metadata(self, memory: MemoryItem) -> tuple[str, str, List[str]]:
        return (
            memory.timestamp,
            memory.memory_type.value,
            memory.tags or []
        )

class MemoryFormatter:
    def __init__(self, indent_size: int = 2, wrap_width: int = 80):
        self.indent_size = indent_size
        self.wrap_width = wrap_width
        self.strategies = {
            Document: LangchainDocumentStrategy(),
            MemoryItem: MemoryItemStrategy()
        }

    def _wrap_text(self, text: str, initial_indent: str, subsequent_indent: str) -> str:
        """Wrap text with proper indentation"""
        return textwrap.fill(
            text,
            width=self.wrap_width,
            initial_indent=initial_indent,
            subsequent_indent=subsequent_indent,
            break_long_words=False,
            break_on_hyphens=False
        )

    def _get_strategy(self, item: Union[Document, MemoryItem]) -> FormattingStrategy:
        """Get the appropriate formatting strategy for the item type"""
        strategy = self.strategies.get(type(item))
        if not strategy:
            raise ValueError(f"Unsupported memory type: {type(item)}")
        return strategy

    def format_single_memory(self, item: Union[Document, MemoryItem], index: int) -> str:
        """Format a single memory entry with numbering"""
        strategy = self._get_strategy(item)

        indent = " " * self.indent_size
        number = f"{index}."

        # Extract content and metadata using the strategy
        content = strategy.get_content(item)
        timestamp, memory_type, tags = strategy.get_metadata(item)
        tags_str = ", ".join(tags) if tags else "none"

        # Create the header with metadata
        header = (
            f"{number} Memory Entry [Type: {memory_type}] "
            f"[Time: {timestamp}]"
        )

        # Format the content with proper wrapping
        content_indent = indent + "   "
        wrapped_content = self._wrap_text(
            content,
            initial_indent=content_indent,
            subsequent_indent=content_indent
        )

        # Format tags on their own line
        tags_line = f"{indent}   Tags: {tags_str}"

        return f"{header}\n{wrapped_content}\n{tags_line}"

    def format_memories(self, items: List[Union[Document, MemoryItem]]) -> str:
        """Format multiple memory entries"""
        if not items:
            return "No memories available."

        formatted_entries = []
        for i, item in enumerate(items, 1):
            formatted_entry = self.format_single_memory(item, i)
            formatted_entries.append(formatted_entry)

        return "\n\n".join(formatted_entries)