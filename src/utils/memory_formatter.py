from langchain.schema import Document
from typing import List
import textwrap


class MemoryFormatter:
    def __init__(self, indent_size: int = 2, wrap_width: int = 80):
        self.indent_size = indent_size
        self.wrap_width = wrap_width
        
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

    def format_single_memory(self, doc: Document, index: int) -> str:
        """Format a single memory entry with numbering"""
        indent = " " * self.indent_size
        number = f"{index}."
        
        # Extract metadata fields with default values
        timestamp = doc.metadata.get('timestamp', 'N/A')
        memory_type = doc.metadata.get('memory_type', 'general')
        memory_category = doc.metadata.get('memory_category', 'uncategorized')
        tags = doc.metadata.get('tags', [])
        tags_str = ", ".join(tags) if tags else "none"
        
        # Create the header with metadata
        header = (
            f"{number} Memory Entry [Type: {memory_type}] "
            f"[Category: {memory_category}] [Time: {timestamp}]"
        )
        
        # Format the content with proper wrapping
        content_indent = indent + "   "
        wrapped_content = self._wrap_text(
            doc.page_content,
            initial_indent=content_indent,
            subsequent_indent=content_indent
        )
        
        # Format tags on their own line
        tags_line = f"{indent}   Tags: {tags_str}"
        
        return f"{header}\n{wrapped_content}\n{tags_line}"

    def format_memories(self, documents: List[Document]) -> str:
        """Format multiple memory entries"""
        if not documents:
            return "No memories available."
            
        formatted_entries = []
        for i, doc in enumerate(documents, 1):
            formatted_entry = self.format_single_memory(doc, i)
            formatted_entries.append(formatted_entry)
            
        return "\n\n".join(formatted_entries)
