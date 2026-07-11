"""Tests that the dry-run path and demo work without heavy dependencies."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def _run(args: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, *args],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=120,
    )


def test_train_dry_run_without_heavy_deps() -> None:
    """`train --dry-run` prints the plan and exits 0 with light deps only."""
    result = _run(
        ["-m", "qlora_tune.train", "--config", "configs/llama-3.1-8b.yaml", "--dry-run"]
    )
    assert result.returncode == 0, result.stderr
    assert "QLORA DRY-RUN TRAINING PLAN" in result.stdout.upper()
    assert "41,943,040" in result.stdout


def test_importing_train_does_not_import_torch() -> None:
    """The training module keeps torch/transformers imports lazy."""
    code = (
        "import sys; import qlora_tune.train, qlora_tune.merge; "
        "banned = {'torch', 'transformers', 'peft', 'trl', 'bitsandbytes'}; "
        "loaded = banned & set(sys.modules); "
        "assert not loaded, f'heavy modules imported eagerly: {loaded}'"
    )
    result = _run(["-c", code])
    assert result.returncode == 0, result.stderr


def test_train_without_dry_run_requires_train_file() -> None:
    """A real run without --train-file fails fast at argument parsing."""
    result = _run(["-m", "qlora_tune.train", "--config", "configs/llama-3.1-8b.yaml"])
    assert result.returncode != 0
    assert "--train-file" in result.stderr


def test_demo_end_to_end(tmp_path: Path) -> None:
    """The full light-path demo runs and produces artifacts and both reports."""
    out = tmp_path / "demo_out"
    result = _run(["-m", "qlora_tune.demo", "--out", str(out)])
    assert result.returncode == 0, result.stderr
    assert "DRY-RUN TRAINING PLAN" in result.stdout
    assert "EVALUATION REPORT" in result.stdout
    for name in ("train.jsonl", "val.jsonl", "test.jsonl"):
        assert (out / name).exists()
