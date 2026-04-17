from __future__ import annotations

from pathlib import Path


def load_judge_prompt(prompt_dir: Path, name: str = "judge_system.txt") -> str:
    return (prompt_dir / name).read_text(encoding="utf-8")
