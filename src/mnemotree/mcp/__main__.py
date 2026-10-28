"""Run the MCP stdio server: python -m mnemotree.mcp --db PATH --namespace NAME."""

from .server import main

raise SystemExit(main())
