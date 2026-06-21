"""Unified diff extraction and validation (output only — no file writes)."""

from __future__ import annotations

import re
from dataclasses import dataclass, field

_DIFF_BLOCK_RE = re.compile(
    r"(?:^|\n)(---\s[^\n]+\n\+\+\+\s[^\n]+\n(?:@@[^\n]+\n(?:[ +\-\\][^\n]*\n?)*)+)",
    re.MULTILINE,
)
_FENCE_DIFF_RE = re.compile(
    r"```(?:diff)?\s*\n(---[^\n]+\n\+\+\+[^\n]+\n(?:@@[^\n]+\n(?:[ +\-\\][^\n]*\n?)*)+)```",
    re.MULTILINE | re.IGNORECASE,
)


@dataclass
class DiffResult:
    raw_output: str
    diffs: list[str] = field(default_factory=list)
    files_touched: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def has_diffs(self) -> bool:
        return bool(self.diffs)


def extract_diffs(text: str) -> DiffResult:
    """Parse unified diffs from model output."""
    found: list[str] = []
    for block in _FENCE_DIFF_RE.findall(text):
        found.append(block.rstrip() + "\n")
    for block in _DIFF_BLOCK_RE.findall(text):
        if block not in found and block.rstrip() + "\n" not in found:
            found.append(block.rstrip() + "\n")

    files: list[str] = []
    warnings: list[str] = []
    for diff in found:
        for line in diff.splitlines():
            if line.startswith("+++ b/"):
                path = line[6:].strip()
                if path != "/dev/null" and path not in files:
                    files.append(path)
            elif line.startswith("--- a/"):
                path = line[6:].strip()
                if path == "/dev/null":
                    continue
        if not _looks_valid(diff):
            warnings.append("One diff block may be incomplete or malformed")

    if not found and "+++" in text and "---" in text:
        warnings.append("Output mentions diff markers but no valid unified diff was parsed")

    return DiffResult(raw_output=text, diffs=found, files_touched=files, warnings=warnings)


def _looks_valid(diff: str) -> bool:
    lines = diff.splitlines()
    has_header = any(l.startswith("--- ") for l in lines) and any(
        l.startswith("+++ ") for l in lines
    )
    has_hunk = any(l.startswith("@@") for l in lines)
    return has_header and has_hunk


def format_diff_report(result: DiffResult) -> str:
    parts: list[str] = []
    if result.diffs:
        parts.append(f"## Diffs ({len(result.diffs)} block(s))")
        if result.files_touched:
            parts.append("Files: " + ", ".join(result.files_touched))
        for index, diff in enumerate(result.diffs, start=1):
            parts.append(f"\n### Diff {index}\n```diff\n{diff.rstrip()}\n```")
    else:
        parts.append("## No unified diffs parsed")
        parts.append("Model output is shown below; check formatting.")
    for warning in result.warnings:
        parts.append(f"\n> Warning: {warning}")
    parts.append("\n## Full model output\n")
    parts.append(result.raw_output)
    return "\n".join(parts)
