"""Pytest bootstrap: make the in-repo ``agent_evals`` package importable."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
