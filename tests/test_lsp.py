import asyncio
import logging
import os
import shutil

import pytest

try:
    import aiofiles

    HAS_AIOFILES = True
except ImportError:
    HAS_AIOFILES = False

from mnemotree.experimental.lsp.manager import LspManager


@pytest.mark.skipif(not HAS_AIOFILES, reason="aiofiles not installed")
@pytest.mark.asyncio
async def test_python_lsp_integration():
    """
    Tests that we can start the Python LSP, open a file, and get symbols.
    """
    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    manager = LspManager(root_dir)

    if shutil.which("pyright-langserver") is None:
        pytest.skip("pyright-langserver not installed")

    logging.getLogger(__name__)

    try:
        logging.info("Starting LSP Manager")
        # Start client
        try:
            client = await manager.get_client("python")
        except RuntimeError as exc:
            pytest.skip(f"pyright-langserver failed to start: {exc}")

        assert client.process is not None

        # Open this very test file (mocking it primarily)
        # We need an actual python content.
        test_file_path = os.path.join(root_dir, "src/mnemotree/core/retrieval.py")
        if not os.path.exists(test_file_path):
            pytest.skip("Test file not found")

        async with aiofiles.open(test_file_path) as f:
            content = await f.read()

        logging.info("Opening document")
        await client.text_document_did_open(test_file_path, content)

        # Request symbols
        # Give it a moment to warm up? LSP is usually fast.
        await asyncio.sleep(1)

        logging.info("Requesting symbols")
        symbols = await client.text_document_symbol(test_file_path)
        logging.info(f"Received {len(symbols) if symbols else 'None'} symbols")
        assert isinstance(symbols, list)
        assert len(symbols) > 0

        # Check for a known symbol in retrieval.py, e.g., 'BaseRetriever' class
        found = False
        for sym in symbols:
            # DocumentSymbol structure: name, kind, range, children
            if sym["name"] == "BaseRetriever":
                found = True
                break
        assert found, "BaseRetriever class symbol not found in retrieval.py symbols"

    finally:
        logging.info("Shutting down LSP Manager")
        await manager.shutdown()
        logging.info("Shutdown complete")
