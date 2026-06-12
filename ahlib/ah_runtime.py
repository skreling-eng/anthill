"""Runtime emulator for .ah programs."""

from __future__ import annotations

import json
import os
import re
import shutil
import sys
import threading
import time
import traceback
from dataclasses import dataclass, field
from itertools import product
from pathlib import Path
from typing import Any, Protocol

from ahlib.ah_actions import (
    ActionExpr,
    CallbackAction,
    ContextAction,
    CustomActionExpr,
    ExternalAction,
    ForAction,
    ZipAction,
    ParallelAction,
    RefAction,
    SequenceAction,
    action_works_with_labels,
    expr_uses_external,
    parse_actions,
)
from ahlib.ah_parser import ARRAY_TYPES, Instruction, ParsedProgram, parse_ah_source
from externals import (
    ExternalContext,
    ExternalInput,
    external_consumes_prompts,
    external_handles_repeat,
    run_external,
    write_invoke,
)

# Externals that may output multiple prompts[] for downstream $image (do not merge after).
_PROMPT_MULTI_OUTPUT_EXTS = frozenset({"texts_to_prompts", "texts2prompts"})

# Minimum seconds between @name[inf] iterations (first iteration runs immediately).
_INF_REPEAT_MIN_INTERVAL = 2.0

# Map changes content_type to array key
class RuntimeCancelled(Exception):
    """Raised when .ah execution is stopped (e.g. app window closing)."""


class ActionCallback(Protocol):
    """Optional progress hooks for external UIs during .ah execution."""

    def action_start(self, action_name: str) -> None: ...

    def action_finish(
        self,
        action_name: str,
        output_context: dict[str, list],
        output_json_path: str | None = None,
        session_base_dir: str | None = None,
    ) -> None: ...

    def action_error(self, action_name: str, error_message: str) -> None: ...


_BINARY_MEDIA_SUFFIXES = frozenset(
    {".wav", ".mp3", ".mp4", ".png", ".jpg", ".jpeg"}
)
_MIN_MEDIA_BYTES = {
    ".wav": 44,
    ".mp3": 4,
    ".mp4": 12,
    ".png": 8,
    ".jpg": 4,
    ".jpeg": 4,
}

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
    "embedding": "embeddings",
    "embeddings": "embeddings",
    "label": "labels",
    "labels": "labels",
}

_IN_MEMORY_ARRAYS = frozenset({"embeddings", "labels"})


@dataclass
class ArrayBundle:
    """Arrays hold only links (paths relative to session root)."""

    prompts: list[str] = field(default_factory=list)
    texts: list[str] = field(default_factory=list)
    images: list[str] = field(default_factory=list)
    sounds: list[str] = field(default_factory=list)
    videos: list[str] = field(default_factory=list)
    files: list[str] = field(default_factory=list)
    embeddings: list[Any] = field(default_factory=list)
    labels: list[Any] = field(default_factory=list)
    changes: list[tuple[str, str, Any]] = field(default_factory=list)

    def as_dict(self) -> dict[str, list]:
        return {
            "prompts": list(self.prompts),
            "texts": list(self.texts),
            "images": list(self.images),
            "sounds": list(self.sounds),
            "videos": list(self.videos),
            "files": list(self.files),
            "embeddings": list(self.embeddings),
            "labels": list(self.labels),
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
            embeddings=list(data.get("embeddings", [])),
            labels=list(data.get("labels", [])),
            changes=changes,
        )

    def copy(self) -> ArrayBundle:
        return ArrayBundle.from_dict(self.as_dict())


class Session:
    def __init__(
        self,
        base_dir: Path,
        *,
        sessions_root: Path | None = None,
        sessions_root_created: bool = False,
    ):
        self.base_dir = base_dir
        self.sessions_root = (sessions_root or base_dir.parent).resolve()
        self.sessions_root_created = sessions_root_created
        self.delete_after_run = False
        self._op_counter = 0
        self._op_lock = threading.Lock()

    def next_op_dir(self, name: str) -> Path:
        with self._op_lock:
            self._op_counter += 1
            counter = self._op_counter
        safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in name)
        op_dir = self.base_dir / f"{counter}_{safe}"
        op_dir.mkdir(parents=True, exist_ok=True)
        return op_dir

    def write_bundle(self, op_dir: Path, bundle: ArrayBundle, label: str) -> None:
        manifest = bundle.as_dict()
        (op_dir / f"{label}.json").write_text(
            json.dumps(manifest, indent=2), encoding="utf-8"
        )
        for array_name in ARRAY_TYPES:
            if array_name == "changes" or array_name in _IN_MEMORY_ARRAYS:
                continue
            for link in getattr(bundle, array_name):
                self._materialize_link(op_dir, link)

    def _materialize_link(self, op_dir: Path, link: str) -> None:
        path = self.base_dir / link
        if path.exists():
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        suffix = path.suffix.lower()
        if suffix in _BINARY_MEDIA_SUFFIXES:
            # Real outputs are written by handlers; never plant empty text blobs.
            return
        placeholder = {
            ".txt": "prompt/text content\n",
        }
        path.write_text(placeholder.get(".txt", ""), encoding="utf-8")

    def resolve_link_path(self, link: str) -> Path:
        from ahlib.link_paths import resolve_link_path

        return resolve_link_path(self, link)

    def ensure_bundle_files_ready(
        self,
        bundle: ArrayBundle,
        *,
        timeout: float = 30.0,
    ) -> None:
        """Wait until bundle-linked files exist and have stable non-trivial size."""
        paths: list[Path] = []
        for key in ARRAY_TYPES:
            if key == "changes" or key in _IN_MEMORY_ARRAYS:
                continue
            for link in getattr(bundle, key):
                path = self.resolve_link_path(link)
                if path.suffix:
                    paths.append(path)

        deadline = time.monotonic() + timeout
        for path in paths:
            min_size = _MIN_MEDIA_BYTES.get(path.suffix.lower(), 1)
            if path.is_absolute() and path.is_file():
                try:
                    if path.stat().st_size >= min_size:
                        continue
                except OSError:
                    pass
            self._wait_for_file_ready(path, min_size=min_size, deadline=deadline)

    def _wait_for_file_ready(
        self, path: Path, *, min_size: int, deadline: float
    ) -> None:
        stable_reads = 0
        last_size = -1
        while time.monotonic() < deadline:
            if path.is_file():
                size = path.stat().st_size
                if size >= min_size and size == last_size:
                    stable_reads += 1
                    if stable_reads >= 2:
                        return
                else:
                    stable_reads = 0
                last_size = size
            else:
                stable_reads = 0
                last_size = -1
            time.sleep(0.05)
        if not path.is_file():
            raise FileNotFoundError(f"Output file not ready: {path}")
        size = path.stat().st_size
        if size < min_size:
            raise FileNotFoundError(
                f"Output file incomplete ({size} bytes, need >= {min_size}): {path}"
            )

    def new_link(self, op_dir: Path, array_name: str, ext: str, content: str | bytes) -> str:
        arr_dir = op_dir / array_name
        arr_dir.mkdir(exist_ok=True)
        existing = len(list(arr_dir.glob(f"*{ext}")))
        filename = f"{existing}{ext}"
        rel = op_dir.relative_to(self.base_dir) / array_name / filename
        path = self.base_dir / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        if isinstance(content, bytes):
            with open(path, "wb") as handle:
                handle.write(content)
                handle.flush()
                os.fsync(handle.fileno())
        else:
            path.write_text(content, encoding="utf-8")
        return str(rel).replace("\\", "/")


def _default_repo_root(session: Session) -> Path:
    parent = session.base_dir.parent
    if parent.name == "sessions":
        return parent.parent.resolve()
    return Path.cwd().resolve()


class Runtime:
    def __init__(
        self,
        program: ParsedProgram,
        session: Session,
        callback: ActionCallback | None = None,
        cancel_event: threading.Event | None = None,
        repo_root: Path | None = None,
    ):
        self.program = program
        self.session = session
        self.callback = callback
        self.cancel_event = cancel_event
        self.repo_root = (
            repo_root.resolve() if repo_root is not None else _default_repo_root(session)
        )
        self._instruction_cache: dict[tuple[str, tuple], ArrayBundle] = {}
        self._session_contexts: dict[str, ArrayBundle] = {}
        self._instruction_call_stack: list[str] = []
        self._last_output_json: Path | None = None

    @property
    def last_output_json_path(self) -> Path | None:
        return self._last_output_json

    def _check_cancelled(self) -> None:
        if self.cancel_event is not None and self.cancel_event.is_set():
            raise RuntimeCancelled("Execution cancelled")

    def _wait_until(self, deadline: float) -> None:
        """Sleep until deadline, checking cancel in small steps."""
        while time.monotonic() < deadline:
            self._check_cancelled()
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            time.sleep(min(0.05, remaining))

    @staticmethod
    def _format_context_action_name(action: ContextAction) -> str:
        if action.scope == "instruction":
            if action.mode == "store":
                return f"%%{action.name}"
            if action.mode == "load":
                return f"{action.name}%%"
            return f"%%{action.name}%%"
        if action.mode == "store":
            return f"%{action.name}"
        if action.mode == "load":
            return f"{action.name}%"
        return f"%{action.name}%"

    @staticmethod
    def _format_error_tail(exc: BaseException) -> str:
        text = "".join(
            traceback.format_exception(type(exc), exc, exc.__traceback__)
        )
        lines = text.rstrip().splitlines()
        return "\n".join(lines[-20:])

    def _notify_action_start(self, action_name: str) -> None:
        if self.callback is not None:
            self.callback.action_start(action_name)

    def _notify_action_finish(
        self,
        action_name: str,
        output: ArrayBundle,
        op_dir: Path | None = None,
    ) -> None:
        self.session.ensure_bundle_files_ready(output)
        output_json_path: str | None = None
        if op_dir is not None:
            path = op_dir / "output.json"
            self._last_output_json = path
            output_json_path = str(path.resolve())
        if self.callback is not None:
            self.callback.action_finish(
                action_name,
                output.as_dict(),
                output_json_path,
                str(self.session.base_dir.resolve()),
            )

    def _notify_action_error(self, action_name: str, exc: BaseException) -> None:
        if self.callback is not None:
            self.callback.action_error(action_name, self._format_error_tail(exc))

    def run(self, entry: str | None = None) -> ArrayBundle:
        self._check_cancelled()
        target = entry or self.program.run_target
        if not target:
            raise ValueError("No run target specified")
        return self._execute_instruction(target, ArrayBundle())

    @staticmethod
    def _freeze_for_cache(value: Any) -> Any:
        if isinstance(value, list):
            return tuple(Runtime._freeze_for_cache(v) for v in value)
        if isinstance(value, dict):
            return tuple(
                sorted(
                    (k, Runtime._freeze_for_cache(v)) for k, v in value.items()
                )
            )
        if isinstance(value, tuple):
            return tuple(Runtime._freeze_for_cache(v) for v in value)
        return value

    @staticmethod
    def _input_key(bundle: ArrayBundle) -> tuple:
        return tuple(
            Runtime._freeze_for_cache(getattr(bundle, key))
            for key in ARRAY_TYPES
            if key != "changes"
        )

    def _execute_instruction(
        self,
        name: str,
        inputs: ArrayBundle,
        *,
        use_cache: bool = True,
        track_progress: bool = True,
    ) -> ArrayBundle:
        action_name = f"@{name}"
        if name not in self.program.instructions:
            if track_progress:
                self._notify_action_start(action_name)
                self._notify_action_error(
                    action_name, KeyError(f"Unknown instruction: @{name}")
                )
            raise KeyError(f"Unknown instruction: @{name}")
        inst = self.program.instructions[name]
        if name in self._instruction_call_stack:
            chain = " -> ".join(
                f"@{n}" for n in self._instruction_call_stack + [name]
            )
            raise RecursionError(
                f"Instruction @{name} is calling itself ({chain}). "
                f"@{name} runs another instruction; %{name} / {name}% is session "
                f"context storage — a separate namespace. Use a different @ name, "
                f"or load context with {name}% instead of @{name}."
            )
        self._instruction_call_stack.append(name)
        try:
            action_expr = parse_actions(inst.actions) if inst.actions else None
            # Never memorize instructions that call $ externals (including via @ refs).
            can_cache = use_cache and not expr_uses_external(
                action_expr, self.program.instructions
            )

            cache_key = (name, self._input_key(inputs))
            if can_cache and cache_key in self._instruction_cache:
                return self._instruction_cache[cache_key].copy()

            self._check_cancelled()

            if track_progress:
                self._notify_action_start(action_name)

            try:
                op_dir = self.session.next_op_dir(name)
                self.session.write_bundle(op_dir, inputs, "input")

                prompt_body = (inst.body or "").strip()

                work = inputs.copy()
                if prompt_body:
                    work = self._compose_instruction_body_into_prompts(
                        inputs, prompt_body, op_dir
                    )

                if inst.actions:
                    instruction_contexts: dict[str, ArrayBundle] = {}
                    work = self._eval_action(
                        action_expr,
                        work,
                        op_dir,
                        instruction_contexts=instruction_contexts,
                    )

                outputs = work.copy()
                outputs = self._apply_changes(outputs, op_dir)
                self.session.write_bundle(op_dir, outputs, "output")

                if can_cache:
                    self._instruction_cache[cache_key] = outputs.copy()
                if track_progress:
                    self._notify_action_finish(action_name, outputs, op_dir)
                return outputs
            except Exception as exc:
                if track_progress:
                    self._notify_action_error(action_name, exc)
                raise
        finally:
            if (
                self._instruction_call_stack
                and self._instruction_call_stack[-1] == name
            ):
                self._instruction_call_stack.pop()

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
        return self._compose_instruction_body_into_prompts(bundle, body, op_dir)

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
        action_name = self._format_context_action_name(action)
        self._notify_action_start(action_name)
        try:
            if action.mode in ("store", "store_load"):
                self._context_store(
                    action.name, action.scope, bundle, instruction_contexts
                )
            if action.mode == "store":
                result = bundle.copy()
            else:
                result = self._context_load(
                    action.name, action.scope, instruction_contexts
                )
            self._notify_action_finish(action_name, result)
            return result
        except Exception as exc:
            self._notify_action_error(action_name, exc)
            raise

    def _eval_custom_action(
        self,
        action: CustomActionExpr,
        bundle: ArrayBundle,
        parent_op_dir: Path,
    ) -> ArrayBundle:
        from ahlib.custom_action_codegen import generate_and_store, run_handler_subprocess

        name = action.name
        action_name = f"&{name}"
        if name not in self.program.custom_actions:
            self._notify_action_start(action_name)
            self._notify_action_error(
                action_name, KeyError(f"Unknown custom action: &{name}")
            )
            raise KeyError(f"Unknown custom action: &{name}")

        spec = (self.program.custom_actions[name].body or "").strip()
        if not spec:
            self._notify_action_start(action_name)
            self._notify_action_error(
                action_name, ValueError(f"&{name}: empty specification")
            )
            raise ValueError(f"&{name}: empty specification")

        self._notify_action_start(action_name)
        op_dir = self.session.next_op_dir(f"_{name}")
        self.session.write_bundle(op_dir, bundle, "input")
        try:
            run_py = generate_and_store(
                name, spec, self.repo_root, op_dir=op_dir
            )
            (op_dir / "custom").mkdir(parents=True, exist_ok=True)
            result = run_handler_subprocess(
                run_py,
                bundle,
                self.session.base_dir,
                op_dir,
                self.repo_root,
            )
            result = self._apply_changes(result, op_dir)
            self.session.write_bundle(op_dir, result, "output")
            self._notify_action_finish(action_name, result, op_dir)
            return result
        except Exception as exc:
            err_path = op_dir / "error.txt"
            err_path.write_text(str(exc) + "\n", encoding="utf-8")
            self._notify_action_error(action_name, exc)
            raise

    def _eval_callback_action(
        self,
        action: CallbackAction,
        bundle: ArrayBundle,
        parent_op_dir: Path,
    ) -> ArrayBundle:
        action_name = f"^{action.name}"
        self._notify_action_start(action_name)
        op_dir = self.session.next_op_dir(action.name)
        self.session.write_bundle(op_dir, bundle, "input")
        try:
            handler = (
                getattr(self.callback, "ah_action", None)
                if self.callback is not None
                else None
            )
            if handler is None:
                raise RuntimeError(
                    f"{action_name} requires a callback with ah_action()"
                )
            inp = ExternalInput(
                bundle=bundle,
                args=dict(action.args),
                prompt_text="",
                repeat=action.repeat or 1,
            )
            result = handler(
                action.name,
                bundle,
                inp,
                dict(action.args),
                repeat=action.repeat or 1,
            )
            if not isinstance(result, ArrayBundle):
                raise TypeError(
                    f"ah_action({action.name!r}) must return ArrayBundle, "
                    f"got {type(result).__name__}"
                )
            self.session.write_bundle(op_dir, result, "output")
            self._notify_action_finish(action_name, result, op_dir)
            return result
        except Exception as exc:
            self._notify_action_error(action_name, exc)
            raise

    def _eval_action(
        self,
        expr: ActionExpr,
        bundle: ArrayBundle,
        parent_op_dir: Path,
        *,
        externals_in_sequence: bool = False,
        instruction_contexts: dict[str, ArrayBundle] | None = None,
    ) -> ArrayBundle:
        self._check_cancelled()
        ctx_store = instruction_contexts if instruction_contexts is not None else {}
        if isinstance(expr, ContextAction):
            return self._eval_context_action(expr, bundle, ctx_store)
        if isinstance(expr, CallbackAction):
            result = self._eval_callback_action(expr, bundle, parent_op_dir)
            return self._propagate_labels_if_needed(bundle, result, expr)
        if isinstance(expr, CustomActionExpr):
            result = self._eval_custom_action(expr, bundle, parent_op_dir)
            return self._propagate_labels_if_needed(bundle, result, expr)
        if isinstance(expr, RefAction):
            use_cache = not externals_in_sequence
            if expr.repeat_infinite:
                action_name = f"@{expr.name}[inf]"
                self._notify_action_start(action_name)
                repeat_op = self.session.next_op_dir(f"{expr.name}_inf")
                self.session.write_bundle(repeat_op, bundle, "input")
                last: ArrayBundle | None = None
                next_start = time.monotonic()
                try:
                    while True:
                        now = time.monotonic()
                        if now < next_start:
                            self._wait_until(next_start)
                        self._check_cancelled()
                        run_result = self._execute_instruction(
                            expr.name,
                            bundle.copy(),
                            use_cache=False,
                            track_progress=False,
                        )
                        last = self._relocate_images_to_op(run_result, repeat_op)
                        next_start = time.monotonic() + _INF_REPEAT_MIN_INTERVAL
                except RuntimeCancelled:
                    if last is not None:
                        last = self._apply_changes(last, repeat_op)
                        self.session.write_bundle(repeat_op, last, "output")
                        self._notify_action_finish(action_name, last, repeat_op)
                    raise
                except Exception as exc:
                    self._notify_action_error(action_name, exc)
                    raise
            if expr.repeat is not None:
                if expr.repeat < 1:
                    return ArrayBundle()
                action_name = f"@{expr.name}[{expr.repeat}]"
                self._notify_action_start(action_name)
                try:
                    repeat_op = self.session.next_op_dir(f"{expr.name}_x{expr.repeat}")
                    self.session.write_bundle(repeat_op, bundle, "input")
                    results: list[ArrayBundle] = []
                    for _ in range(expr.repeat):
                        self._check_cancelled()
                        run_result = self._execute_instruction(
                            expr.name,
                            bundle.copy(),
                            use_cache=False,
                            track_progress=False,
                        )
                        results.append(
                            self._relocate_images_to_op(run_result, repeat_op)
                        )
                    joined = self._join_bundles(results)
                    joined = self._apply_changes(joined, repeat_op)
                    self.session.write_bundle(repeat_op, joined, "output")
                    self._notify_action_finish(action_name, joined, repeat_op)
                    return joined
                except Exception as exc:
                    self._notify_action_error(action_name, exc)
                    raise
            return self._execute_instruction(
                expr.name,
                bundle.copy(),
                use_cache=use_cache,
            )
        if isinstance(expr, ExternalAction):
            result = self._call_external(expr, bundle, parent_op_dir)
            return self._propagate_labels_if_needed(bundle, result, expr)
        if isinstance(expr, ParallelAction):
            return self._eval_parallel(
                expr,
                bundle,
                parent_op_dir,
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
                instruction_contexts=ctx_store,
            )
        raise TypeError(f"Unknown action type: {type(expr)}")

    def _propagate_labels_if_needed(
        self,
        before: ArrayBundle,
        after: ArrayBundle,
        action: ActionExpr,
    ) -> ArrayBundle:
        if action_works_with_labels(action):
            return after
        from ahlib.label_utils import propagate_labels

        return propagate_labels(before, after)

    def _eval_parallel(
        self,
        step: ParallelAction,
        bundle: ArrayBundle,
        parent_op_dir: Path,
        *,
        instruction_contexts: dict[str, ArrayBundle] | None = None,
    ) -> ArrayBundle:
        """Run parallel branches on copies of input; append all arrays into one bundle."""
        work = bundle.copy()
        branch_results: list[ArrayBundle] = []
        for branch in step.branches:
            self._check_cancelled()
            branch_out = self._eval_action(
                branch,
                work.copy(),
                parent_op_dir,
                externals_in_sequence=False,
                instruction_contexts=instruction_contexts,
            )
            branch_out = self._apply_changes(branch_out, parent_op_dir)
            branch_results.append(branch_out)
        joined = self._join_bundles(branch_results, dedupe_images=True)
        return self._propagate_labels_if_needed(bundle, joined, step)

    def _eval_sequence(
        self,
        expr: SequenceAction,
        bundle: ArrayBundle,
        parent_op_dir: Path,
        *,
        externals_in_sequence: bool = False,
        instruction_contexts: dict[str, ArrayBundle] | None = None,
    ) -> ArrayBundle:
        """Evaluate -> steps left to right."""
        current = bundle
        ext_seen = externals_in_sequence
        for step in expr.steps:
            self._check_cancelled()
            current = self._eval_action(
                step,
                current,
                parent_op_dir,
                externals_in_sequence=ext_seen,
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
        if not action_works_with_labels(expr.filter):
            from ahlib.label_utils import propagate_labels

            for_output = propagate_labels(bundle, for_output)

        filtered = self._subtract_bundle_links(initial, for_output)
        body_output = self._eval_action(
            expr.body,
            for_output.copy(),
            for_op,
            externals_in_sequence=True,
            instruction_contexts=instruction_contexts,
        )
        body_output = self._apply_changes(body_output, for_op)
        if not action_works_with_labels(expr.body):
            from ahlib.label_utils import propagate_labels

            body_output = propagate_labels(for_output, body_output)

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
        """zip(arrays){body} or zip(label='name'){body}: run body per slice; join outputs."""
        zip_op = self.session.next_op_dir("zip")
        self.session.write_bundle(zip_op, bundle, "input")

        if expr.label_name:
            from ahlib.label_utils import bundle_from_elements, entries_for_name

            label_entries = entries_for_name(bundle.labels, expr.label_name)
            if not label_entries:
                empty = ArrayBundle()
                self.session.write_bundle(zip_op, empty, "output")
                return empty

            results: list[ArrayBundle] = []
            for _name, elements, _meta in label_entries:
                self._check_cancelled()
                slice_bundle = bundle_from_elements(elements)
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
            self._check_cancelled()
            slice_bundle = ArrayBundle()
            for key in expr.array_keys:
                getattr(slice_bundle, key).append(getattr(bundle, key)[i])
            from ahlib.label_utils import filter_labels_for_bundle

            slice_bundle.labels = filter_labels_for_bundle(bundle.labels, slice_bundle)
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
                elif key == "labels":
                    from ahlib.label_utils import label_entry_key

                    existing = {
                        k
                        for raw in out.labels
                        if (k := label_entry_key(raw)) is not None
                    }
                    for item in items:
                        item_key = label_entry_key(item)
                        if item_key is None:
                            out.labels.append(item)
                            continue
                        if item_key not in existing:
                            out.labels.append(item)
                            existing.add(item_key)
                else:
                    getattr(out, key).extend(items)
        return out

    def _instruction_preserves_multi_prompts(self, ref_name: str) -> bool:
        """True when @ref ends with $texts_to_prompts without join= (N separate prompts)."""
        inst = self.program.instructions.get(ref_name)
        if not inst or not inst.actions:
            return False
        actions = inst.actions
        if "$texts_to_prompts" not in actions and "$texts2prompts" not in actions:
            return False
        if re.search(
            r"texts(?:_to_prompts|2prompts)\s*\([^)]*join\s*=\s*['\"]?(?:1|true|yes)['\"]?",
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

    def _compose_instruction_body_into_prompts(
        self,
        bundle: ArrayBundle,
        body: str,
        op_dir: Path,
    ) -> ArrayBundle:
        """
        Apply @instruction body to prompts[] on a bundle copy.

        1. Body + input prompts with text — concatenate body into every prompt link.
        2. Body + empty prompts — single new prompt link with body only.
        3. No body — pass through input prompts unchanged.
        """
        body = (body or "").strip()
        out = bundle.copy()
        if not body:
            return out
        if self._bundle_has_prompt_text(bundle):
            return self._append_instruction_body_to_prompts(bundle, body, op_dir)
        out.prompts = [
            self.session.new_link(op_dir, "prompts", ".txt", body + "\n")
        ]
        return out

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

    def _bundle_has_prompt_text(self, bundle: ArrayBundle) -> bool:
        """True when prompts[] links contain non-whitespace text."""
        return bool(self._read_prompt_links_text(bundle.prompts))

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
        result = self._execute_instruction(
            ref_name, ArrayBundle(), track_progress=False
        )
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

    @staticmethod
    def _external_arg_variants(arg_lists: dict[str, list[str]]) -> list[dict[str, str]]:
        """Cartesian product of all key=@ref lists; empty when no list args."""
        if not arg_lists:
            return []
        keys = sorted(arg_lists.keys())
        combos = product(*(arg_lists[key] for key in keys))
        return [dict(zip(keys, combo, strict=True)) for combo in combos]

    @staticmethod
    def _branch_external_args(
        action: ExternalAction, variant: dict[str, str]
    ) -> dict[str, str]:
        """Literal args for one fan-out branch (replace @ref values in the AST args)."""
        branch_args = dict(action.args)
        for key, val in variant.items():
            branch_args[key] = val
        return branch_args

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
                if arr_key in _IN_MEMORY_ARRAYS:
                    arr.append(joined)
                else:
                    link = self.session.new_link(op_dir, arr_key, ".txt", joined)
                    arr.append(link)
            elif operation == "del":
                arr[:] = [
                    x
                    for x in arr
                    if x != data
                    and not (isinstance(x, str) and x.endswith(f"/{data}"))
                ]
            elif operation == "add":
                if arr_key in _IN_MEMORY_ARRAYS:
                    arr.append(data)
                elif isinstance(data, str):
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
        args, arg_lists = self._resolve_external_args(action, bundle)
        variants = self._external_arg_variants(arg_lists)
        if len(variants) > 1:
            return self._call_external_arg_fanout(
                action, bundle, op_dir, args, variants
            )
        if variants:
            merged_args = {**args, **variants[0]}
            branch_action = ExternalAction(
                name=action.name,
                args=self._branch_external_args(action, variants[0]),
                repeat=action.repeat,
            )
            return self._invoke_external(branch_action, bundle, op_dir, merged_args, {})
        return self._invoke_external(action, bundle, op_dir, args, arg_lists)

    def _call_external_arg_fanout(
        self,
        action: ExternalAction,
        bundle: ArrayBundle,
        parent_op_dir: Path,
        args: dict[str, str],
        variants: list[dict[str, str]],
    ) -> ArrayBundle:
        """key=@ref list args → parallel ($ext(...), …) with Cartesian product of values."""
        branch_results: list[ArrayBundle] = []
        for variant in variants:
            self._check_cancelled()
            branch_args = {**args, **variant}
            branch_action = ExternalAction(
                name=action.name,
                args=self._branch_external_args(action, variant),
                repeat=action.repeat,
            )
            branch_out = self._invoke_external(
                branch_action,
                bundle.copy(),
                parent_op_dir,
                branch_args,
                {},
            )
            branch_results.append(self._apply_changes(branch_out, parent_op_dir))
        return self._join_bundles(branch_results)

    def _invoke_external(
        self,
        action: ExternalAction,
        bundle: ArrayBundle,
        op_dir: Path,
        args: dict[str, str],
        arg_lists: dict[str, list[str]],
    ) -> ArrayBundle:
        ext_name = action.name
        repeat = action.repeat if action.repeat and action.repeat > 0 else 1
        if repeat > 1 and not external_handles_repeat(ext_name):
            return self._call_external_repeat_fanout(
                action, bundle, op_dir, args, arg_lists, repeat
            )
        return self._invoke_external_once(
            action, bundle, op_dir, args, arg_lists, repeat
        )

    def _call_external_repeat_fanout(
        self,
        action: ExternalAction,
        bundle: ArrayBundle,
        parent_op_dir: Path,
        args: dict[str, str],
        arg_lists: dict[str, list[str]],
        repeat: int,
    ) -> ArrayBundle:
        """$ext(...)[n] when handler ignores repeat → n parallel invocations, joined."""
        branch_results: list[ArrayBundle] = []
        for _ in range(repeat):
            self._check_cancelled()
            branch_action = ExternalAction(
                name=action.name,
                args=action.args,
                repeat=None,
            )
            branch_out = self._invoke_external_once(
                branch_action,
                bundle.copy(),
                parent_op_dir,
                args,
                arg_lists,
                repeat=1,
            )
            branch_results.append(self._apply_changes(branch_out, parent_op_dir))
        return self._join_bundles(branch_results)

    def _invoke_external_once(
        self,
        action: ExternalAction,
        bundle: ArrayBundle,
        op_dir: Path,
        args: dict[str, str],
        arg_lists: dict[str, list[str]],
        repeat: int,
    ) -> ArrayBundle:
        ext_name = action.name
        action_name = f"${ext_name}" if repeat == 1 else f"${ext_name}[{repeat}]"
        op_label = f"${ext_name}" if repeat == 1 else f"${ext_name}_x{repeat}"
        self._notify_action_start(action_name)
        try:
            self._check_cancelled()
            ext_op_dir = self.session.next_op_dir(op_label)
            self.session.write_bundle(ext_op_dir, bundle, "input")

            prompt_text = "\n".join(self._read_prompt_links_text(bundle.prompts))
            ctx = ExternalContext(
                session=self.session,
                op_dir=ext_op_dir,
                cancel_event=self.cancel_event,
                callback=self.callback,
            )
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
            self._notify_action_finish(action_name, out, ext_op_dir)
            return out
        except Exception as exc:
            self._notify_action_error(action_name, exc)
            raise


def create_session_dir(sessions_root: Path) -> Path:
    sessions_root.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    pid = os.getpid()
    # Sub-second suffix avoids collisions when many tests start in the same second.
    unique = time.time_ns() % 1_000_000
    session_dir = sessions_root / f"{stamp}_{pid}_{unique}"
    session_dir.mkdir(parents=True, exist_ok=True)
    return session_dir


def cleanup_session_after_run(session: Session) -> None:
    """Remove session dir; drop empty sessions root if this run created it."""
    if not session.delete_after_run:
        return
    session_dir = session.base_dir.resolve()
    if session_dir.is_dir():
        shutil.rmtree(session_dir)
    sessions_root = session.sessions_root
    if session.sessions_root_created and sessions_root.is_dir():
        try:
            next(sessions_root.iterdir())
        except StopIteration:
            sessions_root.rmdir()


def run_program(
    source: str,
    sessions_root: Path | None = None,
    callback: ActionCallback | None = None,
    cancel_event: threading.Event | None = None,
    repo_root: Path | None = None,
) -> tuple[dict, Path]:
    program = parse_ah_source(source)
    root = (sessions_root or Path("sessions")).resolve()
    created_root = not root.is_dir()
    session_dir = create_session_dir(root)
    session = Session(
        session_dir,
        sessions_root=root,
        sessions_root_created=created_root,
    )
    runtime = Runtime(
        program,
        session,
        callback=callback,
        cancel_event=cancel_event,
        repo_root=repo_root,
    )
    cancelled = False
    try:
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
    except RuntimeCancelled:
        cancelled = True
        raise
    finally:
        from externals.invoke import release_gpu_resources

        release_gpu_resources(reason="run finished")
        if not cancelled:
            cleanup_session_after_run(session)
