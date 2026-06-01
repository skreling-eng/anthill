"""Parser for .ah agentic system language files."""

from __future__ import annotations

import re
import sys
from dataclasses import dataclass, field
from typing import Any


ARRAY_TYPES = (
    "prompts",
    "texts",
    "images",
    "sounds",
    "videos",
    "files",
    "embeddings",
    "labels",
    "changes",
)


@dataclass
class Instruction:
    name: str
    actions: str | None = None
    body: str = ""


@dataclass
class RunCommand:
    target: str


@dataclass
class ParsedProgram:
    instructions: dict[str, Instruction] = field(default_factory=dict)
    run_target: str | None = None


_HEADER_RE = re.compile(r"^@(\w+)\s*(?::\s*(.*))?$")
_BLOCK_COMMENT_RE = re.compile(r"/\*.*?\*/", re.DOTALL)


def strip_block_comments(source: str) -> str:
    """Remove /* ... */ blocks (non-greedy, multiline) before parsing."""
    return _BLOCK_COMMENT_RE.sub("", source)


def _is_comment_line(line: str) -> bool:
    """Lines starting with # (after optional whitespace) are comments."""
    return line.lstrip().startswith("#")


def _strip_line(line: str) -> str:
    """Trim trailing whitespace; comment lines should be skipped by the caller."""
    return line.rstrip()


def _delimiter_depth(s: str, open_ch: str, close_ch: str) -> int:
    """Net depth of open_ch/close_ch, ignoring delimiters inside '...' or \"...\"."""
    depth = 0
    in_sq = False
    in_dq = False
    for ch in s:
        if in_sq:
            if ch == "'":
                in_sq = False
            continue
        if in_dq:
            if ch == '"':
                in_dq = False
            continue
        if ch == "'":
            in_sq = True
        elif ch == '"':
            in_dq = True
        elif ch == open_ch:
            depth += 1
        elif ch == close_ch:
            depth -= 1
    return depth


def _brace_depth(s: str) -> int:
    return _delimiter_depth(s, "{", "}")


def _paren_depth(s: str) -> int:
    return _delimiter_depth(s, "(", ")")


def _actions_need_continuation(s: str) -> bool:
    """True while action text has unclosed { } or ( )."""
    return _brace_depth(s) > 0 or _paren_depth(s) > 0


def _strip_outer_action_braces(actions: str) -> str:
    """Remove wrapping { ... } only when braces enclose the whole action string."""
    s = actions.strip()
    if not (s.startswith("{") and s.endswith("}")):
        return actions
    depth = 0
    for i, ch in enumerate(s):
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0 and i != len(s) - 1:
                return actions
    return s[1:-1].strip()


def _read_braced_block(lines: list[str], start_idx: int) -> tuple[str, int]:
    """Collect lines from start_idx until braces balance (opening `{` on first line)."""
    parts: list[str] = []
    depth = 0
    i = start_idx
    while i < len(lines):
        if _is_comment_line(lines[i]):
            i += 1
            continue
        line = lines[i]
        for ch in line:
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
        parts.append(line)
        i += 1
        if depth <= 0:
            break
    return "\n".join(parts), i


def _read_triple_quoted_body(lines: list[str], start_idx: int) -> tuple[str, int]:
    first = lines[start_idx].strip()
    if not first.startswith('"""'):
        raise ValueError(f"Expected triple-quoted body at line {start_idx + 1}")
    content_parts: list[str] = []
    rest = first[3:]
    if rest.endswith('"""') and len(rest) > 3:
        return rest[:-3], start_idx + 1
    if rest:
        content_parts.append(rest)
    i = start_idx + 1
    while i < len(lines):
        line = lines[i]
        if '"""' in line:
            before, _, _after = line.partition('"""')
            if before:
                content_parts.append(before)
            return "\n".join(content_parts), i + 1
        content_parts.append(line)
        i += 1
    raise ValueError("Unclosed triple-quoted body")


def parse_ah_source(source: str) -> ParsedProgram:
    """Parse .ah file content into a program dictionary in memory."""
    source = strip_block_comments(source)
    lines = source.splitlines()
    program = ParsedProgram()
    i = 0
    n = len(lines)

    while i < n:
        if _is_comment_line(lines[i]):
            i += 1
            continue

        line = _strip_line(lines[i]).strip()
        if not line:
            i += 1
            continue

        if line.startswith("run "):
            target = line[4:].strip()
            if target.startswith("@"):
                target = target[1:]
            program.run_target = target
            i += 1
            continue

        m = _HEADER_RE.match(line)
        if not m:
            i += 1
            continue

        name, actions_part = m.group(1), m.group(2)
        actions: str | None = None
        body = ""
        i += 1

        if actions_part is not None:
            actions_part = actions_part.strip()
            if actions_part.startswith("{"):
                block, i = _read_braced_block(lines, i - 1)
                block = block.strip()
                brace = block.find("{")
                if brace >= 0:
                    block = block[brace:]
                actions = _strip_outer_action_braces(block)
            else:
                actions = actions_part
                combined = [actions]
                while _actions_need_continuation("\n".join(combined)) and i < n:
                    if _is_comment_line(lines[i]):
                        i += 1
                        continue
                    combined.append(_strip_line(lines[i]).strip())
                    i += 1
                actions = "\n".join(combined).strip()
                if actions_part.strip().startswith("{"):
                    actions = _strip_outer_action_braces(actions)

        if i < n and not _is_comment_line(lines[i]) and lines[i].strip().startswith('"""'):
            body, i = _read_triple_quoted_body(lines, i)
        else:
            body_lines: list[str] = []
            while i < n:
                if _is_comment_line(lines[i]):
                    i += 1
                    continue
                peek = _strip_line(lines[i]).strip()
                if not peek:
                    i += 1
                    if body_lines:
                        break
                    continue
                if peek.startswith("@") or peek.startswith("run "):
                    break
                body_lines.append(_strip_line(lines[i]))
                i += 1
            body = "\n".join(body_lines).strip()

        if name in program.instructions:
            print(
                f"warning: @{name} redefined; earlier definition is replaced",
                file=sys.stderr,
                flush=True,
            )
        program.instructions[name] = Instruction(name=name, actions=actions, body=body)

    return program


def program_to_dict(program: ParsedProgram) -> dict[str, Any]:
    """Simple dictionary representation for in-memory use."""
    return {
        "instructions": {
            name: {
                "actions": inst.actions,
                "body": inst.body,
            }
            for name, inst in program.instructions.items()
        },
        "run": program.run_target,
    }


def parse_ah_file(path: str) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        source = f.read()
    return program_to_dict(parse_ah_source(source))
