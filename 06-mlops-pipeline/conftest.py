"""Repo-root conftest so ``mlops_pipeline`` is importable during tests."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
