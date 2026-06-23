"""Shared prompt-file loader for the controller component."""

from pathlib import Path

_PROMPTS_DIR = Path(__file__).resolve().parent / "prompts"


def load_prompt(filename: str) -> str:
    """Read a prompt template from the package ``prompts/`` directory."""
    return (_PROMPTS_DIR / filename).read_text(encoding="utf-8").strip()
