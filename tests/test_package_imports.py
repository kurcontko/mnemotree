"""Importing the package stays light: optional extras load only when used."""

import os
import subprocess
import sys
from pathlib import Path

import mnemotree

SRC = str(Path(mnemotree.__file__).parents[1])
PROBE = """
import sys, mnemotree, mnemotree.mcp.server
optional = ('fastapi', 'uvicorn', 'httpx', 'mnemotree.service.app')
loaded = [m for m in optional if m in sys.modules]
assert not loaded, loaded
"""


def test_importing_the_package_loads_no_optional_extra():
    probe = subprocess.run(
        [sys.executable, "-c", PROBE],
        env={**os.environ, "PYTHONPATH": SRC},
        capture_output=True,
        text=True,
        check=False,
    )
    assert probe.returncode == 0, probe.stderr
