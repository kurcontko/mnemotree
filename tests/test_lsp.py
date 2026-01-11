import asyncio
import os
import pytest
import logging
from mnemotree.experimental.lsp.manager import LspManager

@pytest.mark.asyncio
async def test_python_lsp_integration():
    """
    Tests that we can start the Python LSP, open a file, and get symbols.
    """
    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    manager = LspManager(root_dir)
    
    logger = logging.getLogger(__name__)
    
    try:
        logging.info("Starting LSP Manager")
        # Start client
        client = await manager.get_client("python")
        assert client.process is not None
        
        # Open this very test file (mocking it primarily)
        # We need an actual python content.
        test_file_path = os.path.join(root_dir, "src/mnemotree/core/retrieval.py")
        if not os.path.exists(test_file_path):
            pytest.skip("Test file not found")
            
        with open(test_file_path, "r") as f:
            content = f.read()
            
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
