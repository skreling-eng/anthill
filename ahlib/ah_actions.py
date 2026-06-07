"""Action expression parser for .ah instruction pipelines."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Union

# AST nodes
@dataclass
class RefAction:
    name: str
    repeat: int | None = None  # @name[n] — run n times and join results
    repeat_infinite: bool = False  # @name[inf] — repeat until cancel; ≥2s between iterations


@dataclass
class ExternalAction:
    name: str
    args: dict[str, str]
    repeat: int | None = None  # $name(...)[n] — n output variants


@dataclass
class ParallelAction:
    branches: list["ActionExpr"]


@dataclass
class SequenceAction:
    steps: list["ActionExpr"]


@dataclass
class ForAction:
    """for(filter){ body } — partition input, process matched items in body, rejoin."""

    filter: ActionExpr
    body: ActionExpr


@dataclass
class ZipAction:
    """zip(images, texts){ body } or zip(label='name'){ body }."""

    array_keys: list[str]
    body: ActionExpr
    label_name: str | None = None


@dataclass
class ContextAction:
    """Context storage step: %name (store), name% (load), %name% (store+load)."""

    name: str
    mode: str  # "store", "load", "store_load"
    scope: str  # "session" or "instruction"


@dataclass
class CallbackAction:
    """UI callback step: ^name — handled by runtime callback.ah_action()."""

    name: str
    args: dict[str, str]
    repeat: int | None = None


@dataclass
class CustomActionExpr:
    """Custom &name step — runs generated Python from custom_actions/."""

    name: str


ActionExpr = Union[
    RefAction,
    ExternalAction,
    ParallelAction,
    SequenceAction,
    ForAction,
    ZipAction,
    ContextAction,
    CallbackAction,
    CustomActionExpr,
]

_REF_RE = re.compile(r"^@(\w+)(?:\[(inf|\d+)\])?$", re.IGNORECASE)
_EXTERNAL_RE = re.compile(r"^\$(\w+)(?:\((.*)\))?(?:\[(\d+)\])?$")
_CALLBACK_RE = re.compile(r"^\^(\w+)(?:\((.*)\))?(?:\[(\d+)\])?$")
_CUSTOM_RE = re.compile(r"^&(\w+)$")
_CONTEXT_RE = re.compile(
    r"^(?P<prefix>%+)?(?P<name>\w+)(?P<suffix>%+)?$"
)


def _parse_context_token(tok: str) -> ContextAction | None:
    """Parse %ccc, ccc%, %ccc%, %%ccc, ccc%%, %%ccc%%."""
    m = _CONTEXT_RE.match(tok)
    if not m or (not m.group("prefix") and not m.group("suffix")):
        return None
    prefix = m.group("prefix") or ""
    suffix = m.group("suffix") or ""
    name = m.group("name")
    if prefix and not prefix.replace("%", ""):
        if len(prefix) not in (1, 2) or (suffix and len(suffix) not in (1, 2)):
            raise ValueError(f"Invalid context token: {tok}")
        scope = "instruction" if len(prefix) == 2 else "session"
        if suffix:
            if len(suffix) != len(prefix):
                raise ValueError(f"Invalid context token: {tok}")
            mode = "store_load"
        else:
            mode = "store"
    elif suffix and not prefix:
        if len(suffix) not in (1, 2):
            raise ValueError(f"Invalid context token: {tok}")
        scope = "instruction" if len(suffix) == 2 else "session"
        mode = "load"
    else:
        raise ValueError(f"Invalid context token: {tok}")
    return ContextAction(name=name, mode=mode, scope=scope)


def _parse_zip_keys(keys_src: str) -> tuple[list[str], str | None]:
    """Return (array_keys, label_name) for zip(...) spec."""
    stripped = keys_src.strip()
    args = _parse_args(stripped)
    if "label" in args and "," not in stripped:
        return [], args["label"]
    array_keys = [k.strip() for k in stripped.split(",") if k.strip()]
    return array_keys, None


def _parse_args(arg_str: str) -> dict[str, str]:
    if not arg_str.strip():
        return {}
    not_m = re.match(
        r"^not\s+('([^']*)'|\"([^\"]*)\"|(\w+))\s*$",
        arg_str.strip(),
        re.IGNORECASE,
    )
    if not_m:
        return {"not": not_m.group(2) or not_m.group(3) or not_m.group(4) or ""}
    args: dict[str, str] = {}
    for part in re.findall(
        r"(\w+)\s*=\s*('[^']*'|\"[^\"]*\"|\[[^\]]*\]|@\w+|\w+)", arg_str
    ):
        key, val = part
        if (val.startswith("'") and val.endswith("'")) or (
            val.startswith('"') and val.endswith('"')
        ):
            val = val[1:-1]
        args[key] = val
    quoted = [
        m.group(1) or m.group(2)
        for m in re.finditer(r"'([^']*)'|\"([^\"]*)\"", arg_str.strip())
    ]
    if quoted and "_path" not in args:
        args["_path"] = quoted[0]
        if len(quoted) >= 2 and "_path2" not in args and "pattern" not in args:
            args["_path2"] = quoted[1]
    if not args and arg_str.strip():
        if re.fullmatch(r"[\w,\s]+", arg_str.strip()):
            args["_arrays"] = arg_str.strip()
    return args


def _read_balanced(
    source: str, pos: int, open_ch: str, close_ch: str
) -> tuple[str, int]:
    """Read from pos (after open_ch) until matching close_ch at depth 0."""
    depth = 1
    start = pos
    while pos < len(source):
        ch = source[pos]
        if ch == open_ch:
            depth += 1
        elif ch == close_ch:
            depth -= 1
            if depth == 0:
                return source[start:pos], pos + 1
        pos += 1
    raise ValueError(f"Unclosed {open_ch} in action expression")


def _read_zip_block(source: str, i: int) -> tuple[str, str, int]:
    """source[i] is 'z' in 'zip'. Returns (array_keys_src, body_src, index after block)."""
    if source[i : i + 3] != "zip":
        raise ValueError("expected zip(")
    pos = i + 3
    while pos < len(source) and source[pos].isspace():
        pos += 1
    if pos >= len(source) or source[pos] != "(":
        raise ValueError("zip( expected")
    pos += 1
    keys_src, pos = _read_balanced(source, pos, "(", ")")
    while pos < len(source) and source[pos].isspace():
        pos += 1
    if pos >= len(source) or source[pos] != "{":
        raise ValueError("{ expected after zip(...)")
    pos += 1
    body_src, pos = _read_balanced(source, pos, "{", "}")
    return keys_src.strip(), body_src.strip(), pos


def _read_for_block(source: str, i: int) -> tuple[str, str, int]:
    """source[i] is 'f' in 'for'. Returns (filter_src, body_src, index after block)."""
    if source[i : i + 3] != "for":
        raise ValueError("expected for(")
    pos = i + 3
    while pos < len(source) and source[pos].isspace():
        pos += 1
    if pos >= len(source) or source[pos] != "(":
        raise ValueError("for( expected")
    pos += 1
    filter_src, pos = _read_balanced(source, pos, "(", ")")
    while pos < len(source) and source[pos].isspace():
        pos += 1
    if pos >= len(source) or source[pos] != "{":
        raise ValueError("{ expected after for(...)")
    pos += 1
    body_src, pos = _read_balanced(source, pos, "{", "}")
    return filter_src.strip(), body_src.strip(), pos


def _tokenize_actions(source: str) -> list[str]:
    """Split action string into tokens preserving structure."""
    source = " ".join(source.split())
    tokens: list[str] = []
    i = 0
    while i < len(source):
        if source[i].isspace():
            i += 1
            continue
        if source[i : i + 2] == "->":
            tokens.append("->")
            i += 2
            while i < len(source) and source[i].isspace():
                i += 1
            if source[i : i + 2] == "->":
                i += 2
            continue
        if source[i : i + 3] == "for" and (
            i + 3 >= len(source) or source[i + 3] in "( \t"
        ):
            filter_src, body_src, end = _read_for_block(source, i)
            tokens.append(f"for({filter_src}){{{body_src}}}")
            i = end
            continue
        if source[i : i + 3] == "zip" and (
            i + 3 >= len(source) or source[i + 3] in "( \t"
        ):
            keys_src, body_src, end = _read_zip_block(source, i)
            tokens.append(f"zip({keys_src}){{{body_src}}}")
            i = end
            continue
        if source[i] in ",()":
            tokens.append(source[i])
            i += 1
            continue
        if source[i] == "%":
            j = i
            while j < len(source) and source[j] == "%":
                j += 1
            while j < len(source) and (source[j].isalnum() or source[j] == "_"):
                j += 1
            while j < len(source) and source[j] == "%":
                j += 1
            tok = source[i:j]
            if _parse_context_token(tok) is None:
                raise ValueError(f"Invalid context token: {tok}")
            tokens.append(tok.strip())
            i = j
            continue
        if source[i].isalnum() or source[i] == "_":
            j = i
            while j < len(source) and (source[j].isalnum() or source[j] == "_"):
                j += 1
            if j < len(source) and source[j] == "%":
                k = j
                while k < len(source) and source[k] == "%":
                    k += 1
                tok = source[i:k]
                if _parse_context_token(tok) is not None:
                    tokens.append(tok.strip())
                    i = k
                    continue
        if source[i] in "@$^&":
            j = i + 1
            depth = 0
            while j < len(source):
                ch = source[j]
                if ch == "(":
                    depth += 1
                elif ch == ")":
                    if depth > 0:
                        depth -= 1
                        j += 1
                        if depth == 0 and source[i] == "$":
                            while j < len(source) and source[j].isspace():
                                j += 1
                            if j < len(source) and source[j] == "[":
                                k = j + 1
                                while k < len(source) and source[k].isdigit():
                                    k += 1
                                if k < len(source) and source[k] == "]":
                                    j = k + 1
                            break
                        continue
                    break
                elif ch in ",->" and depth == 0:
                    break
                j += 1
            tokens.append(source[i:j].strip())
            i = j
            continue
        i += 1
    return tokens


def _parse_expr(tokens: list[str], pos: int = 0) -> tuple[ActionExpr, int]:
    if pos >= len(tokens):
        raise ValueError("Unexpected end of action expression")

    if tokens[pos] == "(":
        branches: list[ActionExpr] = []
        pos += 1
        while pos < len(tokens) and tokens[pos] != ")":
            branch, pos = _parse_expr(tokens, pos)
            branches.append(branch)
            if pos < len(tokens) and tokens[pos] == ",":
                pos += 1
        if pos >= len(tokens) or tokens[pos] != ")":
            raise ValueError("Unclosed parenthesis in action expression")
        node: ActionExpr = ParallelAction(branches=branches)
        pos += 1
    else:
        tok = tokens[pos]
        if tok.startswith("for(") and tok.endswith("}"):
            m_for = re.match(r"^for\((.*)\)\{(.*)\}$", tok, re.DOTALL)
            if not m_for:
                raise ValueError(f"Invalid for block: {tok}")
            filter_expr = parse_actions(m_for.group(1))
            body_expr = parse_actions(m_for.group(2))
            if filter_expr is None or body_expr is None:
                raise ValueError("for(...) filter and body must be non-empty")
            node = ForAction(filter=filter_expr, body=body_expr)
            pos += 1
            steps = [node]
            while pos < len(tokens) and tokens[pos] == "->":
                pos += 1
                nxt, pos = _parse_expr(tokens, pos)
                steps.append(nxt)
            if len(steps) == 1:
                return steps[0], pos
            return SequenceAction(steps=steps), pos
        if tok.startswith("zip(") and tok.endswith("}"):
            m_zip = re.match(r"^zip\((.*)\)\{(.*)\}$", tok, re.DOTALL)
            if not m_zip:
                raise ValueError(f"Invalid zip block: {tok}")
            array_keys, label_name = _parse_zip_keys(m_zip.group(1))
            if not array_keys and not label_name:
                raise ValueError(
                    "zip(...) requires array names or label='name', "
                    "e.g. zip(images, texts) or zip(label='good')"
                )
            body_expr = parse_actions(m_zip.group(2))
            if body_expr is None:
                raise ValueError("zip(...) body must be non-empty")
            node = ZipAction(
                array_keys=array_keys,
                body=body_expr,
                label_name=label_name,
            )
            pos += 1
            steps = [node]
            while pos < len(tokens) and tokens[pos] == "->":
                pos += 1
                nxt, pos = _parse_expr(tokens, pos)
                steps.append(nxt)
            if len(steps) == 1:
                return steps[0], pos
            return SequenceAction(steps=steps), pos
        ctx = _parse_context_token(tok)
        if ctx is not None:
            node = ctx
        elif (m_cb := _CALLBACK_RE.match(tok)):
            args = _parse_args(m_cb.group(2) or "")
            repeat = int(m_cb.group(3)) if m_cb.group(3) else None
            node = CallbackAction(name=m_cb.group(1), args=args, repeat=repeat)
        elif (m_custom := _CUSTOM_RE.match(tok)):
            node = CustomActionExpr(name=m_custom.group(1))
        elif (m_ref := _REF_RE.match(tok)):
            suffix = m_ref.group(2)
            if suffix is None:
                repeat: int | None = None
                repeat_infinite = False
            elif suffix.lower() == "inf":
                repeat = None
                repeat_infinite = True
            else:
                repeat = int(suffix)
                repeat_infinite = False
            node = RefAction(
                name=m_ref.group(1),
                repeat=repeat,
                repeat_infinite=repeat_infinite,
            )
        else:
            m_ext = _EXTERNAL_RE.match(tok)
            if not m_ext:
                raise ValueError(f"Unknown action token: {tok}")
            args = _parse_args(m_ext.group(2) or "")
            repeat = int(m_ext.group(3)) if m_ext.group(3) else None
            node = ExternalAction(name=m_ext.group(1), args=args, repeat=repeat)
        pos += 1

    steps = [node]
    while pos < len(tokens) and tokens[pos] == "->":
        pos += 1
        nxt, pos = _parse_expr(tokens, pos)
        steps.append(nxt)

    if len(steps) == 1:
        return steps[0], pos
    return SequenceAction(steps=steps), pos


def parse_actions(source: str | None) -> ActionExpr | None:
    if not source or not source.strip():
        return None
    tokens = _tokenize_actions(source)
    expr, pos = _parse_expr(tokens, 0)
    if pos != len(tokens):
        raise ValueError(f"Unexpected tokens after position {pos}: {tokens[pos:]}")
    return expr


def action_works_with_labels(expr: ActionExpr) -> bool:
    """True when the action manages labels[] itself (skip auto-propagation)."""
    if isinstance(expr, ExternalAction):
        from externals import external_works_with_labels

        return external_works_with_labels(expr.name)
    if isinstance(expr, ZipAction):
        return expr.label_name is not None
    if isinstance(expr, SequenceAction):
        return False
    if isinstance(expr, ParallelAction):
        return False
    if isinstance(expr, ForAction):
        return False
    return False


def action_starts_with_external(expr: ActionExpr | None) -> bool:
    """True if the first step of this action tree is a $ external call."""
    if expr is None:
        return False
    if isinstance(expr, ExternalAction):
        return True
    if isinstance(expr, SequenceAction):
        return bool(expr.steps) and isinstance(expr.steps[0], ExternalAction)
    if isinstance(expr, ForAction):
        return False
    if isinstance(expr, ZipAction):
        return False
    if isinstance(expr, ParallelAction):
        return any(isinstance(b, ExternalAction) for b in expr.branches)
    return False


def expr_uses_external(
    expr: ActionExpr | None,
    instructions: dict | None = None,
    _visiting: set[str] | None = None,
) -> bool:
    """True if this action subtree calls any $ external (including via @ refs)."""
    if expr is None:
        return False
    if isinstance(expr, ExternalAction):
        return True
    if isinstance(expr, RefAction):
        if not instructions:
            return False
        if expr.name in (_visiting or ()):
            return False
        inst = instructions.get(expr.name)
        if inst is None:
            return False
        visiting = set(_visiting or ())
        visiting.add(expr.name)
        sub = parse_actions(getattr(inst, "actions", None))
        return expr_uses_external(sub, instructions, visiting)
    if isinstance(expr, ForAction):
        return expr_uses_external(expr.filter, instructions, _visiting) or expr_uses_external(
            expr.body, instructions, _visiting
        )
    if isinstance(expr, ZipAction):
        return expr_uses_external(expr.body, instructions, _visiting)
    if isinstance(expr, ParallelAction):
        return any(expr_uses_external(b, instructions, _visiting) for b in expr.branches)
    if isinstance(expr, SequenceAction):
        return any(expr_uses_external(s, instructions, _visiting) for s in expr.steps)
    return False
