"""Post-process generated .ah scripts from $code."""

from __future__ import annotations

import re

_RUN_LINE = re.compile(r"^\s*run\s+@(\w+)\s*$", re.MULTILINE | re.IGNORECASE)
_INSTR_LINE = re.compile(r"^@(\w+)\s*:", re.MULTILINE)
_PREFERRED_ENTRIES = ("answer", "main", "run", "gen", "chat")


def _entry_instruction_name(script: str) -> str | None:
    names = _INSTR_LINE.findall(script)
    if not names:
        return None
    for preferred in _PREFERRED_ENTRIES:
        if preferred in names:
            return preferred
    return names[-1]


def ensure_run_line(script: str) -> tuple[str, bool]:
    """
    Append ``run @name`` when the model omitted it.

    Returns (text, changed).
    """
    text = script.rstrip() + "\n" if script.strip() else ""
    if not text.strip():
        return script, False
    if _RUN_LINE.search(text):
        return text if text.endswith("\n") else text + "\n", False
    entry = _entry_instruction_name(text)
    if not entry:
        return script, False
    fixed = text + f"run @{entry}\n"
    return fixed, True


def fixup_generated_ah(script: str) -> tuple[str, list[str]]:
    """Apply safe fixes to model output. Returns (text, notes)."""
    notes: list[str] = []
    text, added_run = ensure_run_line(script)
    if added_run:
        entry = _entry_instruction_name(script) or "?"
        notes.append(f"appended missing run @{entry}")
    return text, notes
