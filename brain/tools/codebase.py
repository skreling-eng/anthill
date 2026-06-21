"""Codebase indexing and file reading for AH language projects."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from brain.config import BrainConfig, load_config

_BLOCKED_PARTS = frozenset(
    {".cache", ".git", ".venv", ".venvs", "__pycache__", "node_modules"}
)


def is_blocked_path(rel_path: str) -> bool:
    """True when a relative path should never be indexed or read."""
    rel = rel_path.strip().replace("\\", "/").lstrip("/")
    if not rel:
        return True
    return bool(set(rel.split("/")) & _BLOCKED_PARTS)


@dataclass(frozen=True)
class FileEntry:
    rel_path: str
    size: int
    kind: str  # ah | py | md | other


class CodebaseIndex:
    """Walk the codebase and expose tree + search helpers."""

    def __init__(self, config: BrainConfig | None = None):
        self.config = config or load_config()
        self.root = self.config.codebase_root
        self._entries: list[FileEntry] | None = None

    def refresh(self) -> None:
        self._entries = None

    def entries(self) -> list[FileEntry]:
        if self._entries is not None:
            return self._entries
        out: list[FileEntry] = []
        skip = self.config.skip_dirs
        exts = set(self.config.code_extensions)
        special = set(self.config.special_names)
        max_files = self.config.max_files_in_tree

        for path in sorted(self.root.rglob("*")):
            if len(out) >= max_files:
                break
            if not path.is_file():
                continue
            rel = path.relative_to(self.root).as_posix()
            parts = set(rel.split("/"))
            if parts & skip:
                continue
            if path.name.startswith(".") and path.name not in special:
                continue
            if path.suffix not in exts and path.name not in special:
                continue
            try:
                size = path.stat().st_size
            except OSError:
                continue
            kind = _file_kind(path)
            out.append(FileEntry(rel_path=rel, size=size, kind=kind))
        self._entries = out
        return out

    def tree_lines(self, *, max_lines: int = 120) -> list[str]:
        entries = self.entries()
        by_dir: dict[str, list[str]] = {}
        for entry in entries:
            p = Path(entry.rel_path)
            parent = p.parent.as_posix() if p.parent != Path(".") else "."
            by_dir.setdefault(parent, []).append(p.name)
        lines: list[str] = []
        for directory in sorted(by_dir):
            lines.append(f"{directory}/")
            for name in sorted(by_dir[directory]):
                lines.append(f"  {name}")
            if len(lines) >= max_lines:
                lines.append(f"  ... ({len(entries) - max_lines} more files)")
                break
        return lines

    def tree_summary(self) -> str:
        entries = self.entries()
        kinds: dict[str, int] = {}
        for e in entries:
            kinds[e.kind] = kinds.get(e.kind, 0) + 1
        kind_line = ", ".join(f"{k}={v}" for k, v in sorted(kinds.items()))
        return (
            f"Root: {self.root}\n"
            f"Files indexed: {len(entries)} ({kind_line})\n"
            + "\n".join(self.tree_lines())
        )

    def resolve(self, rel_path: str) -> Path:
        rel = rel_path.strip().replace("\\", "/").lstrip("/")
        if is_blocked_path(rel):
            raise ValueError(f"Blocked path: {rel_path}")
        path = (self.root / rel).resolve()
        if not str(path).startswith(str(self.root)):
            raise ValueError(f"Path escapes codebase root: {rel_path}")
        return path

    def read(self, rel_path: str) -> str:
        return read_file(self.resolve(rel_path), max_bytes=self.config.max_file_bytes)

    def grep(self, pattern: str, *, limit: int = 30) -> list[tuple[str, int, str]]:
        rx = re.compile(pattern, re.IGNORECASE)
        hits: list[tuple[str, int, str]] = []
        for entry in self.entries():
            if len(hits) >= limit:
                break
            try:
                text = self.read(entry.rel_path)
            except (OSError, ValueError):
                continue
            for lineno, line in enumerate(text.splitlines(), start=1):
                if rx.search(line):
                    hits.append((entry.rel_path, lineno, line.strip()[:200]))
                    if len(hits) >= limit:
                        break
        return hits


def read_file(path: Path, *, max_bytes: int) -> str:
    if not path.is_file():
        raise FileNotFoundError(str(path))
    size = path.stat().st_size
    if size > max_bytes:
        head = path.read_bytes()[:max_bytes]
        text = head.decode("utf-8", errors="replace")
        return text + f"\n\n[... truncated {size - max_bytes} bytes ...]"
    return path.read_text(encoding="utf-8", errors="replace")


def search_files(index: CodebaseIndex, query: str, *, limit: int = 20) -> list[FileEntry]:
    """Simple filename + content keyword search."""
    q = query.strip().lower()
    if not q:
        return []
    terms = [t for t in re.split(r"\W+", q) if len(t) >= 2]
    scored: list[tuple[int, FileEntry]] = []
    for entry in index.entries():
        score = 0
        name = entry.rel_path.lower()
        for term in terms:
            if term in name:
                score += 3
        if score == 0:
            try:
                body = index.read(entry.rel_path).lower()
            except (OSError, ValueError):
                continue
            for term in terms:
                if term in body:
                    score += 1
        if score > 0:
            scored.append((score, entry))
    scored.sort(key=lambda x: (-x[0], x[1].rel_path))
    return [e for _, e in scored[:limit]]


def _file_kind(path: Path) -> str:
    if path.suffix == ".ah":
        return "ah"
    if path.suffix == ".py":
        return "py"
    if path.suffix == ".md" or path.name.endswith("_description"):
        return "md"
    return "other"
