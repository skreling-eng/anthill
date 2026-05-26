"""Runtime emulator for .ah programs."""

from __future__ import annotations

import json
import os
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ahlib.ah_actions import (
    ActionExpr,
    ContextAction,
    ExternalAction,
    ForAction,
    ZipAction,
    ParallelAction,
    RefAction,
    SequenceAction,
    action_starts_with_external,
    expr_uses_external,
    parse_actions,
)
from ahlib.ah_parser import ARRAY_TYPES, Instruction, ParsedProgram, parse_ah_source
from externals import (
    ExternalContext,
    ExternalInput,
    external_consumes_prompts,
    run_external,
    write_invoke,
)

# Externals that may output multiple prompts[] for downstream $image (do not merge after).
_PROMPT_MULTI_OUTPUT_EXTS = frozenset({"texts_to_prompts"})

# Map changes content_type to array key
_CHANGE_TYPE_MAP = {
    "prompt": "prompts",
    "prompts": "prompts",
    "text": "texts",
    "texts": "texts",
    "image": "images",
    "images": "images",
    "sound": "sounds",
    "sounds": "sounds",
    "video": "videos",
    "videos": "videos",
    "file": "files",
    "files": "files",
}


@dataclass
class ArrayBundle:
    """Arrays hold only links (paths relative to session root)."""

    prompts: list[str] = field(default_factory=list)
    texts: list[str] = field(default_factory=list)
    images: list[str] = field(default_factory=list)
    sounds: list[str] = field(default_factory=list)
    videos: list[str] = field(default_factory=list)
    files: list[str] = field(default_factory=list)
    changes: list[tuple[str, str, Any]] = field(default_factory=list)

    def as_dict(self) -> dict[str, list]:
        return {
            "prompts": list(self.prompts),
            "texts": list(self.texts),
            "images": list(self.images),
            "sounds": list(self.sounds),
            "videos": list(self.videos),
            "files": list(self.files),
            "changes": [list(c) for c in self.changes],
        }

    @classmethod
    def from_dict(cls, data: dict[str, list]) -> ArrayBundle:
        changes = [tuple(c) for c in data.get("changes", [])]
        return cls(
            prompts=list(data.get("prompts", [])),
            texts=list(data.get("texts", [])),
            images=list(data.get("images", [])),
            sounds=list(data.get("sounds", [])),
            videos=list(data.get("videos", [])),
            files=list(data.get("files", [])),
            changes=changes,
        )

    def copy(self) -> ArrayBundle:
        return ArrayBundle.from_dict(self.as_dict())


class Session:
    def __init__(self, base_dir: Path):
        self.base_dir = base_dir
        self._op_counter = 0

    def next_op_dir(self, name: str) -> Path:
        self._op_counter += 1
        safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in name)
        op_dir = self.base_dir / f"{self._op_counter}_{safe}"
        op_dir.mkdir(parents=True, exist_ok=True)
        return op_dir

    def write_bundle(self, op_dir: Path, bundle: ArrayBundle, label: str) -> None:
        manifest = bundle.as_dict()
        (op_dir / f"{label}.json").write_text(
            json.dumps(manifest, indent=2), encoding="utf-8"
        )
        for array_name in ARRAY_TYPES:
            if array_name == "changes":
                continue
            for link in getattr(bundle, array_name):
                self._materialize_link(op_dir, link)

    def _materialize_link(self, op_dir: Path, link: str) -> None:
        path = self.base_dir / link
        if path.exists():
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        suffix = path.suffix.lower()
        placeholder = {
            ".txt": "prompt/text content\n",
            ".png": b"\x89PNG\r\n\x1a\n",
            ".jpg": b"\xff\xd8\xff",
            ".jpeg": b"\xff\xd8\xff",
            ".mp3": b"ID3",
            ".mp4": b"\x00\x00\x00\x1cftyp",
        }
        if suffix in (".png", ".jpg", ".jpeg", ".mp3", ".mp4"):
            path.write_bytes(placeholder.get(suffix, b""))
        else:
            path.write_text(placeholder.get(".txt", ""), encoding="utf-8")

    def new_link(self, op_dir: Path, array_name: str, ext: str, content: str | bytes) -> str:
        arr_dir = op_dir / array_name
        arr_dir.mkdir(exist_ok=True)
        existing = len(list(arr_dir.glob(f"*{ext}")))
        filename = f"{existing}{ext}"
        rel = op_dir.relative_to(self.base_dir) / array_name / filename
        path = self.base_dir / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        if isinstance(content, bytes):
            path.write_bytes(content)
        else:
            path.write_text(content, encoding="utf-8")
        return str(rel).replace("\\", "/")


class Runtime:
    def __init__(self, program: ParsedProgram, session: Session):
        self.program = program
        self.session = session
        self._instruction_cache: dict[tuple[str, tuple], ArrayBundle] = {}
        self._session_contexts: dict[str, ArrayBundle] = {}

    def run(self, entry: str | None = None) -> ArrayBundle:
        target = entry or self.program.run_target
        if not target:
            raise ValueError("No run target specified")
        return self._execute_instruction(target, ArrayBundle())

    @staticmethod
    def _input_key(bundle: ArrayBundle) -> tuple:
        return tuple(
            tuple(getattr(bundle, key))
            for key in ARRAY_TYPES
            if key != "changes"
        )

    def _execute_instruction(
        self, name: str, inputs: ArrayBundle, *, use_cache: bool = True
    ) -> ArrayBundle:
        if name not in self.program.instructions:
            raise KeyError(f"Unknown instruction: @{name}")
        inst = self.program.instructions[name]
        action_expr = parse_actions(inst.actions) if inst.actions else None
        # Never memorize instructions that call $ externals (including via @ refs).
        can_cache = use_cache and not expr_uses_external(
            action_expr, self.program.instructions
        )

        cache_key = (name, self._input_key(inputs))
        if can_cache and cache_key in self._instruction_cache:
            return self._instruction_cache[cache_key].copy()

        op_dir = self.session.next_op_dir(name)
        self.session.write_bundle(op_dir, inputs, "input")

        work = inputs.copy()
        body_prepended = False

        # Body before actions when it is the input prompt for $ externals (e.g. @x: $llm[10]).
        if (
            inst.body
            and inst.actions
            and not inputs.prompts
            and action_starts_with_external(action_expr)
        ):
            link = self.session.new_link(
                op_dir, "prompts", ".txt", inst.body + "\n"
            )
            work.prompts.append(link)
            body_prepended = True

        # @image: @good_quality -> $image — merge body before $, not after (prompt consumed).
        pending_instruction_body: list[str] = []
        if (
            inst.body
            and inst.actions
            and not body_prepended
            and not action_starts_with_external(action_expr)
        ):
            pending_instruction_body.append(inst.body)
            body_prepended = True

        if inst.actions:
            instruction_contexts: dict[str, ArrayBundle] = {}
            work = self._eval_action(
                action_expr,
                work,
                op_dir,
                instruction_body_pending=pending_instruction_body or None,
                instruction_contexts=instruction_contexts,
            )

        if inst.body and pending_instruction_body:
            body = pending_instruction_body[0]
            if work.prompts:
                work = self._append_instruction_body_to_prompts(
                    work, body, op_dir
                )
            else:
                link = self.session.new_link(op_dir, "prompts", ".txt", body + "\n")
                work.prompts.append(link)
            pending_instruction_body.clear()
        elif inst.body and not body_prepended:
            if work.prompts:
                work = self._append_instruction_body_to_prompts(
                    work, inst.body, op_dir
                )
            else:
                link = self.session.new_link(
                    op_dir, "prompts", ".txt", inst.body + "\n"
                )
                work.prompts.append(link)

        outputs = work.copy()
        outputs = self._apply_changes(outputs, op_dir)
        self.session.write_bundle(op_dir, outputs, "output")

        if can_cache:
            self._instruction_cache[cache_key] = outputs.copy()
        return outputs

    def _merge_pending_instruction_body(
        self,
        bundle: ArrayBundle,
        pending: list[str],
        op_dir: Path,
    ) -> ArrayBundle:
        """Join @instruction body with prompts[] before a prompt-consuming $."""
        if not pending:
            return bundle
        body = pending[0]
        pending.clear()
        if bundle.prompts:
            merged = self._compose_prompt_body(bundle, body, op_dir)
            return self._apply_changes(merged, op_dir)
        out = bundle.copy()
        link = self.session.new_link(op_dir, "prompts", ".txt", body + "\n")
        out.prompts.append(link)
        return out

    @staticmethod
    def _bundle_is_empty(bundle: ArrayBundle) -> bool:
        return all(not getattr(bundle, key) for key in ARRAY_TYPES)

    def _context_store(
        self,
        name: str,
        scope: str,
        incoming: ArrayBundle,
        instruction_contexts: dict[str, ArrayBundle],
    ) -> None:
        store = (
            self._session_contexts
            if scope == "session"
            else instruction_contexts
        )
        if name in store:
            store[name] = self._join_bundles([store[name], incoming.copy()])
        else:
            store[name] = incoming.copy()

    def _context_load(
        self,
        name: str,
        scope: str,
        instruction_contexts: dict[str, ArrayBundle],
    ) -> ArrayBundle:
        store = (
            self._session_contexts
            if scope == "session"
            else instruction_contexts
        )
        if name not in store or self._bundle_is_empty(store[name]):
            print(
                f"warning: context {name!r} is empty",
                flush=True,
                file=sys.stderr,
            )
            return ArrayBundle()
        return store[name].copy()

    def _eval_context_action(
        self,
        action: ContextAction,
        bundle: ArrayBundle,
        instruction_contexts: dict[str, ArrayBundle],
    ) -> ArrayBundle:
        if action.mode in ("store", "store_load"):
            self._context_store(
                action.name, action.scope, bundle, instruction_contexts
            )
        if action.mode == "store":
            return bundle.copy()
        return self._context_load(action.name, action.scope, instruction_contexts)

    def _eval_action(
        self,
        expr: ActionExpr,
        bundle: ArrayBundle,
        parent_op_dir: Path,
        *,
        externals_in_sequence: bool = False,
        instruction_body_pending: list[str] | None = None,
        instruction_contexts: dict[str, ArrayBundle] | None = None,
    ) -> ArrayBundle:
        pending = instruction_body_pending
        ctx_store = instruction_contexts if instruction_contexts is not None else {}
        if isinstance(expr, ContextAction):
            return self._eval_context_action(expr, bundle, ctx_store)
        if isinstance(expr, RefAction):
            use_cache = not externals_in_sequence
            if expr.repeat is not None:
                if expr.repeat < 1:
                    return ArrayBundle()
                repeat_op = self.session.next_op_dir(f"{expr.name}_x{expr.repeat}")
                self.session.write_bundle(repeat_op, bundle, "input")
                results: list[ArrayBundle] = []
                for _ in range(expr.repeat):
                    run_result = self._execute_instruction(
                        expr.name, bundle.copy(), use_cache=False
                    )
                    results.append(self._relocate_images_to_op(run_result, repeat_op))
                joined = self._join_bundles(results)
                joined = self._apply_changes(joined, repeat_op)
                self.session.write_bundle(repeat_op, joined, "output")
                return joined
            return self._execute_instruction(
                expr.name, bundle.copy(), use_cache=use_cache
            )
        if isinstance(expr, ExternalAction):
            if pending and external_consumes_prompts(expr.name):
                bundle = self._merge_pending_instruction_body(
                    bundle, pending, parent_op_dir
                )
            return self._call_external(expr, bundle, parent_op_dir)
        if isinstance(expr, ParallelAction):
            return self._eval_parallel(
                expr,
                bundle,
                parent_op_dir,
                pending=pending,
                instruction_contexts=ctx_store,
            )
        if isinstance(expr, ForAction):
            return self._eval_for_action(
                expr,
                bundle,
                parent_op_dir,
                externals_in_sequence=externals_in_sequence,
                instruction_contexts=ctx_store,
            )
        if isinstance(expr, ZipAction):
            return self._eval_zip_action(
                expr,
                bundle,
                parent_op_dir,
                externals_in_sequence=externals_in_sequence,
                instruction_contexts=ctx_store,
            )
        if isinstance(expr, SequenceAction):
            return self._eval_sequence(
                expr,
                bundle,
                parent_op_dir,
                externals_in_sequence=externals_in_sequence,
                instruction_body_pending=pending,
                instruction_contexts=ctx_store,
            )
        raise TypeError(f"Unknown action type: {type(expr)}")

    def _eval_parallel(
        self,
        step: ParallelAction,
        bundle: ArrayBundle,
        parent_op_dir: Path,
        *,
        pending: list[str] | None = None,
        instruction_contexts: dict[str, ArrayBundle] | None = None,
    ) -> ArrayBundle:
        """Run parallel branches on copies of input; append all arrays into one bundle."""
        work = bundle.copy()
        branch_pending = pending
        if pending and any(
            isinstance(b, ExternalAction) and external_consumes_prompts(b.name)
            for b in step.branches
        ):
            work = self._merge_pending_instruction_body(
                bundle.copy(), pending, parent_op_dir
            )
            branch_pending = None
        branch_results: list[ArrayBundle] = []
        for branch in step.branches:
            branch_out = self._eval_action(
                branch,
                work.copy(),
                parent_op_dir,
                externals_in_sequence=False,
                instruction_body_pending=branch_pending,
                instruction_contexts=instruction_contexts,
            )
            branch_out = self._apply_changes(branch_out, parent_op_dir)
            branch_results.append(branch_out)
        return self._join_bundles(branch_results, dedupe_images=True)

    def _eval_sequence(
        self,
        expr: SequenceAction,
        bundle: ArrayBundle,
        parent_op_dir: Path,
        *,
        externals_in_sequence: bool = False,
        instruction_body_pending: list[str] | None = None,
        instruction_contexts: dict[str, ArrayBundle] | None = None,
    ) -> ArrayBundle:
        """Evaluate -> steps left to right."""
        current = bundle
        pending = instruction_body_pending
        ext_seen = externals_in_sequence
        for step in expr.steps:
            if (
                pending
                and isinstance(step, ExternalAction)
                and external_consumes_prompts(step.name)
            ):
                current = self._merge_pending_instruction_body(
                    current, pending, parent_op_dir
                )
            current = self._eval_action(
                step,
                current,
                parent_op_dir,
                externals_in_sequence=ext_seen,
                instruction_body_pending=pending,
                instruction_contexts=instruction_contexts,
            )
            if expr_uses_external(step, self.program.instructions):
                ext_seen = True
            current = self._apply_changes(current, parent_op_dir)
            if not isinstance(step, ParallelAction) and not isinstance(step, ZipAction):
                current = self._join_prompts_after_step(current, step, parent_op_dir)
        return current

    def _relocate_images_to_op(
        self, bundle: ArrayBundle, target_op: Path
    ) -> ArrayBundle:
        """Copy image files into target_op so each repeat run gets a new file."""
        if not bundle.images:
            return bundle
        out = bundle.copy()
        new_images: list[str] = []
        for img_link in bundle.images:
            src = self.session.base_dir / img_link
            content = src.read_bytes() if src.exists() else b""
            new_images.append(
                self.session.new_link(target_op, "images", ".png", content)
            )
        out.images = new_images
        return out

    def _subtract_bundle_links(
        self, base: ArrayBundle, remove: ArrayBundle
    ) -> ArrayBundle:
        """Remove from base every link that appears in remove (any array type)."""
        out = base.copy()
        for key in ARRAY_TYPES:
            if key == "changes":
                continue
            drop = set(getattr(remove, key))
            arr = getattr(out, key)
            arr[:] = [x for x in arr if x not in drop]
        return out

    def _eval_for_action(
        self,
        expr: ForAction,
        bundle: ArrayBundle,
        parent_op_dir: Path,
        *,
        externals_in_sequence: bool,
        instruction_contexts: dict[str, ArrayBundle] | None = None,
    ) -> ArrayBundle:
        """for(filter){body}: split input, correct matched items, rejoin with the rest."""
        for_op = self.session.next_op_dir("for")
        self.session.write_bundle(for_op, bundle, "input")

        initial = bundle.copy()
        for_output = self._eval_action(
            expr.filter,
            bundle.copy(),
            for_op,
            externals_in_sequence=externals_in_sequence,
            instruction_contexts=instruction_contexts,
        )
        for_output = self._apply_changes(for_output, for_op)

        filtered = self._subtract_bundle_links(initial, for_output)
        body_output = self._eval_action(
            expr.body,
            for_output.copy(),
            for_op,
            externals_in_sequence=True,
            instruction_contexts=instruction_contexts,
        )
        body_output = self._apply_changes(body_output, for_op)

        result = self._join_bundles([filtered, body_output])
        result = self._apply_changes(result, for_op)
        self.session.write_bundle(for_op, result, "output")
        return result

    def _eval_zip_action(
        self,
        expr: ZipAction,
        bundle: ArrayBundle,
        parent_op_dir: Path,
        *,
        externals_in_sequence: bool,
        instruction_contexts: dict[str, ArrayBundle] | None = None,
    ) -> ArrayBundle:
        """zip(arrays){body}: run body on each index slice; join outputs."""
        zip_op = self.session.next_op_dir("zip")
        self.session.write_bundle(zip_op, bundle, "input")

        for key in expr.array_keys:
            if key not in ARRAY_TYPES or key == "changes":
                raise ValueError(f"zip(...) invalid array name: {key!r}")

        lengths = [len(getattr(bundle, key)) for key in expr.array_keys]
        if not lengths or min(lengths) == 0:
            empty = ArrayBundle()
            self.session.write_bundle(zip_op, empty, "output")
            return empty

        if len(set(lengths)) != 1:
            print(
                f"warning: zip({', '.join(expr.array_keys)}) array lengths differ "
                f"{dict(zip(expr.array_keys, lengths))}; using shortest",
                flush=True,
                file=sys.stderr,
            )
        count = min(lengths)

        results: list[ArrayBundle] = []
        for i in range(count):
            slice_bundle = ArrayBundle()
            for key in expr.array_keys:
                getattr(slice_bundle, key).append(getattr(bundle, key)[i])
            body_out = self._eval_action(
                expr.body,
                slice_bundle,
                zip_op,
                externals_in_sequence=True,
                instruction_contexts=instruction_contexts,
            )
            body_out = self._apply_changes(body_out, zip_op)
            results.append(body_out)

        joined = self._join_bundles(results)
        joined = self._apply_changes(joined, zip_op)
        self.session.write_bundle(zip_op, joined, "output")
        return joined

    def _join_bundles(
        self, bundles: list[ArrayBundle], *, dedupe_images: bool = False
    ) -> ArrayBundle:
        out = ArrayBundle()
        for b in bundles:
            for key in ARRAY_TYPES:
                items = getattr(b, key)
                if dedupe_images and key == "images":
                    for item in items:
                        if item not in out.images:
                            out.images.append(item)
                else:
                    getattr(out, key).extend(items)
        return out

    def _instruction_preserves_multi_prompts(self, ref_name: str) -> bool:
        """True when @ref ends with $texts_to_prompts without join= (N separate prompts)."""
        inst = self.program.instructions.get(ref_name)
        if not inst or not inst.actions:
            return False
        actions = inst.actions
        if "$texts_to_prompts" not in actions:
            return False
        if re.search(
            r"texts_to_prompts\s*\([^)]*join\s*=\s*['\"]?(?:1|true|yes)['\"]?",
            actions,
            re.IGNORECASE,
        ):
            return False
        return True

    def _instruction_is_body_only(self, ref_name: str) -> bool:
        """True for @name with body text and no action pipeline."""
        inst = self.program.instructions.get(ref_name)
        if not inst or not inst.body:
            return False
        return not (inst.actions or "").strip()

    def _join_prompts_after_step(
        self, bundle: ArrayBundle, step: ActionExpr, op_dir: Path
    ) -> ArrayBundle:
        """After a -> step, merge prompts[] into one link (same as @ body compose)."""
        if isinstance(step, ExternalAction):
            if step.name in _PROMPT_MULTI_OUTPUT_EXTS:
                return bundle
        elif isinstance(step, RefAction):
            if self._instruction_preserves_multi_prompts(step.name):
                return bundle
            if self._instruction_is_body_only(step.name):
                return bundle
            if len(bundle.prompts) > 1:
                return bundle
            return self._consolidate_prompts(bundle, op_dir)
        return bundle

    def _consolidate_prompts(self, bundle: ArrayBundle, op_dir: Path) -> ArrayBundle:
        """Merge multiple prompts[] links into a single joined file."""
        if len(bundle.prompts) <= 1:
            return bundle
        sources = list(bundle.prompts)
        for link in sources:
            bundle.changes.append(("prompt", "del", link))
        bundle.changes.append(("prompt", "join", {"sources": sources, "text": ""}))
        return self._apply_changes(bundle, op_dir)

    def _compose_prompt_body(
        self, bundle: ArrayBundle, body: str, op_dir: Path
    ) -> ArrayBundle:
        """Merge body with existing prompts via changes (del + join), not array append."""
        if bundle.prompts:
            sources = list(bundle.prompts)
            for link in sources:
                bundle.changes.append(("prompt", "del", link))
            bundle.changes.append(
                ("prompt", "join", {"sources": sources, "text": body})
            )
        else:
            link = self.session.new_link(op_dir, "prompts", ".txt", body + "\n")
            bundle.prompts.append(link)
        return bundle

    def _append_instruction_body_to_prompts(
        self, bundle: ArrayBundle, body: str, op_dir: Path
    ) -> ArrayBundle:
        """Append @instruction body to each prompts[] link, or create one if empty."""
        body = body.strip()
        if not bundle.prompts:
            link = self.session.new_link(op_dir, "prompts", ".txt", body + "\n")
            out = bundle.copy()
            out.prompts.append(link)
            return out
        out = bundle.copy()
        out.prompts = []
        for link in bundle.prompts:
            text = ""
            path = self.session.base_dir / link
            if path.exists():
                text = path.read_text(encoding="utf-8", errors="replace").strip()
            if text and body:
                combined = f"{text}\n{body}\n"
            elif body:
                combined = body + "\n"
            else:
                combined = text + ("\n" if text else "")
            out.prompts.append(
                self.session.new_link(op_dir, "prompts", ".txt", combined)
            )
        return out

    def _read_prompt_links_text(self, links: list[str]) -> list[str]:
        parts: list[str] = []
        for link in links:
            path = self.session.base_dir / link
            if path.exists():
                text = path.read_text(encoding="utf-8", errors="replace").strip()
                if text:
                    parts.append(text)
        return parts

    def _read_bundle_texts(self, bundle: ArrayBundle) -> list[str]:
        texts: list[str] = []
        for link in bundle.texts:
            path = self.session.base_dir / link
            if path.exists():
                text = path.read_text(encoding="utf-8", errors="replace").strip()
                if text:
                    texts.append(text)
        return texts

    def _resolve_instruction_texts(self, ref_name: str) -> list[str]:
        """Run @ref on empty inputs; return all strings from its output texts array."""
        result = self._execute_instruction(ref_name, ArrayBundle())
        return self._read_bundle_texts(result)

    def _resolve_external_args(
        self, action: ExternalAction, bundle: ArrayBundle
    ) -> tuple[dict[str, str], dict[str, list[str]]]:
        """Split $ext(...) args into scalars and lists (key=@instruction)."""
        args: dict[str, str] = {}
        arg_lists: dict[str, list[str]] = {}
        for key, val in action.args.items():
            if val.startswith("@"):
                ref_name = val[1:]
                items = self._resolve_instruction_texts(ref_name)
                if not items:
                    raise ValueError(
                        f"@{ref_name} produced no texts for "
                        f"${action.name}({key}=@{ref_name})"
                    )
                arg_lists[key] = items
            else:
                args[key] = val
        return args, arg_lists

    def _apply_changes(self, bundle: ArrayBundle, op_dir: Path) -> ArrayBundle:
        if not bundle.changes:
            return bundle
        pending = list(bundle.changes)
        joins = [c for c in pending if c[1] == "join"]
        rest = [c for c in pending if c[1] != "join"]
        for content_type, operation, data in joins + rest:
            arr_key = _CHANGE_TYPE_MAP.get(content_type, content_type)
            if arr_key not in ARRAY_TYPES:
                continue
            arr = getattr(bundle, arr_key)
            if operation == "join":
                sources = data.get("sources", []) if isinstance(data, dict) else []
                extra = data.get("text", "") if isinstance(data, dict) else ""
                parts = self._read_prompt_links_text(sources)
                if extra.strip():
                    parts.append(extra.strip())
                joined = "\n".join(parts) + ("\n" if parts else "")
                link = self.session.new_link(op_dir, arr_key, ".txt", joined)
                arr.append(link)
            elif operation == "del":
                arr[:] = [x for x in arr if x != data and not x.endswith(f"/{data}")]
            elif operation == "add":
                if isinstance(data, str):
                    link = self.session.new_link(
                        op_dir, arr_key, Path(data).suffix or ".txt", data
                    )
                    arr.append(link)
                else:
                    arr.append(str(data))
        bundle.changes.clear()
        return bundle

    def _call_external(
        self, action: ExternalAction, bundle: ArrayBundle, op_dir: Path
    ) -> ArrayBundle:
        ext_name = action.name
        repeat = action.repeat if action.repeat and action.repeat > 0 else 1
        op_label = f"${ext_name}" if repeat == 1 else f"${ext_name}_x{repeat}"
        ext_op_dir = self.session.next_op_dir(op_label)
        self.session.write_bundle(ext_op_dir, bundle, "input")

        prompt_text = "\n".join(self._read_prompt_links_text(bundle.prompts))
        args, arg_lists = self._resolve_external_args(action, bundle)
        ctx = ExternalContext(session=self.session, op_dir=ext_op_dir)
        inp = ExternalInput(
            bundle=bundle,
            args=args,
            prompt_text=prompt_text,
            repeat=repeat,
            arg_lists=arg_lists,
        )
        write_invoke(ext_op_dir, inp)
        out = run_external(ext_name, ctx, inp)

        out = self._apply_changes(out, ext_op_dir)
        if external_consumes_prompts(ext_name):
            out.prompts.clear()
        self.session.write_bundle(ext_op_dir, out, "output")
        return out


def create_session_dir(sessions_root: Path) -> Path:
    sessions_root.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    pid = os.getpid()
    # Sub-second suffix avoids collisions when many tests start in the same second.
    unique = time.time_ns() % 1_000_000
    session_dir = sessions_root / f"{stamp}_{pid}_{unique}"
    session_dir.mkdir(parents=True, exist_ok=True)
    return session_dir


def run_program(source: str, sessions_root: Path | None = None) -> tuple[dict, Path]:
    program = parse_ah_source(source)
    root = sessions_root or Path("sessions")
    session_dir = create_session_dir(root)
    runtime = Runtime(program, Session(session_dir))
    result = runtime.run()
    meta = {
        "session": str(session_dir),
        "run": program.run_target,
        "output": result.as_dict(),
    }
    (session_dir / "session.json").write_text(
        json.dumps(meta, indent=2), encoding="utf-8"
    )
    return meta, session_dir
