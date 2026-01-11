from __future__ import annotations

import logging
import shutil
from typing import Literal

from .client import LspClient

logger = logging.getLogger(__name__)

LanguageId = Literal["python"]


class LspManager:
    """
    Manages language server instances for different languages.
    """

    def __init__(self, root_dir: str):
        self.root_dir = root_dir
        self.clients: dict[LanguageId, LspClient] = {}

    async def get_client(self, language_id: LanguageId) -> LspClient:
        """Gets or starts an LSP client for the given language."""
        if language_id in self.clients:
            return self.clients[language_id]

        if language_id == "python":
            client = await self._start_python_ls()
            self.clients["python"] = client
            return client
        
        raise ValueError(f"Unsupported language: {language_id}")

    async def _start_python_ls(self) -> LspClient:
        # Check for pyright-langserver
        executable = shutil.which("pyright-langserver")
        if not executable:
            # Fallback to checking if 'pyright' module exposes it (common in uv/pip installs)
            # But usually it puts `pyright-langserver` in bin.
            # If not found, raise.
             raise RuntimeError("pyright-langserver not found. Please install `pyright`.")
        
        cmd = [executable, "--stdio"]
        client = LspClient(cmd, self.root_dir)
        await client.start()
        
        # Initialize
        await client.initialize()
        return client

    async def shutdown(self) -> None:
        """Shuts down all active clients."""
        for client in self.clients.values():
            await client.stop()
        self.clients.clear()
