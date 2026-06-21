"""Agent orchestrator — plan, gather context, generate diffs."""

from __future__ import annotations

import json
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Callable

from brain.agent.prompts import GENERATOR_SYSTEM, LANG_REFERENCE, PLANNER_SYSTEM
from brain.config import BrainConfig, load_config
from brain.conversation.history import ConversationStore, HistoryTurn
from brain.llm.code_model import CodeModel, get_code_model
from brain.llm.context_limit import AgentPromptParts, ChatHistory, trim_plain_text
from brain.tools.codebase import CodebaseIndex, is_blocked_path, search_files
from brain.tools.diff import DiffResult, extract_diffs, format_diff_report
from brain.tools.externals_catalog import (
    format_catalog_markdown,
    is_externals_catalog_query,
    scan_externals,
)
from brain.tools.web_search import web_search

StatusCallback = Callable[[str], None]


@dataclass
class AgentResult:
    request: str
    plan: dict
    context_files: dict[str, str] = field(default_factory=dict)
    grep_hits: list[tuple[str, int, str]] = field(default_factory=list)
    search_results: list[dict[str, str]] = field(default_factory=list)
    diff_result: DiffResult | None = None
    report: str = ""
    session_dir: Path | None = None
    elapsed_s: float = 0.0
    error: str | None = None
    trim_notes: list[str] = field(default_factory=list)


class AgentOrchestrator:
    """Multi-step agent: analyze request → read code → search web → emit diffs."""

    def __init__(
        self,
        config: BrainConfig | None = None,
        model: CodeModel | None = None,
        on_status: StatusCallback | None = None,
    ):
        self.config = config or load_config()
        self.model = model or get_code_model(self.config)
        self.index = CodebaseIndex(self.config)
        self.on_status = on_status or (lambda _msg: None)
        self._cancel = threading.Event()

    def cancel(self) -> None:
        self._cancel.set()

    def _status(self, msg: str) -> None:
        self.on_status(msg)

    def _check_cancel(self) -> None:
        if self._cancel.is_set():
            raise RuntimeError("Cancelled")

    def run(
        self,
        request: str,
        *,
        conversation: list[HistoryTurn] | None = None,
    ) -> AgentResult:
        self._cancel.clear()
        started = time.monotonic()
        request = request.strip()
        prior = list(conversation or [])
        chat_history = ConversationStore.to_chat_pairs(
            prior[-self.config.max_conversation_turns :]
        )
        result = AgentResult(request=request, plan={})
        session_dir = self._new_session()
        result.session_dir = session_dir

        try:
            self._check_cancel()
            if is_externals_catalog_query(request) and not prior:
                self._finish_catalog(request, result, session_dir)
                self._status("Done")
                result.elapsed_s = time.monotonic() - started
                return result

            self._check_cancel()
            self._status("Indexing codebase…")
            tree = self.index.tree_summary()

            self._check_cancel()
            self._status("Planning changes…")
            plan = self._plan(request, tree, chat_history)
            plan = self._enrich_plan_from_conversation(plan, prior)
            result.plan = plan
            self._write_json(session_dir / "plan.json", plan)

            self._check_cancel()
            self._status("Reading source files…")
            context_files, grep_hits = self._gather_files(plan)
            result.context_files = context_files
            result.grep_hits = grep_hits
            self._write_json(
                session_dir / "context_index.json",
                {"files": list(context_files.keys()), "grep_hits": grep_hits},
            )

            self._check_cancel()
            queries = plan.get("search_queries") or []
            if queries:
                self._status("Searching the web…")
            search_results = self._gather_search(queries)
            result.search_results = search_results
            if search_results:
                self._write_json(session_dir / "search.json", search_results)

            self._check_cancel()
            self._status("Generating diffs…")
            raw = self._generate(
                request,
                plan,
                context_files,
                grep_hits,
                search_results,
                tree,
                chat_history,
            )
            result.trim_notes = list(self.model.last_trim_notes)
            if result.trim_notes:
                self._write_json(session_dir / "trim_notes.json", result.trim_notes)
            diff_result = extract_diffs(raw)
            result.diff_result = diff_result
            result.report = format_diff_report(diff_result)
            (session_dir / "output.txt").write_text(raw, encoding="utf-8")
            (session_dir / "report.md").write_text(result.report, encoding="utf-8")
            if diff_result.diffs:
                (session_dir / "diffs.patch").write_text(
                    "\n".join(diff_result.diffs), encoding="utf-8"
                )

            self._status("Done")
        except Exception as exc:
            result.error = str(exc)
            result.report = f"**Error:** {exc}"
            self._status(f"Error: {exc}")
            if session_dir:
                (session_dir / "error.txt").write_text(str(exc), encoding="utf-8")

        result.elapsed_s = time.monotonic() - started
        return result

    def _finish_catalog(
        self, request: str, result: AgentResult, session_dir: Path
    ) -> None:
        """Fast path: scan externals/*/_description without LLM."""
        self._status("Scanning externals catalog…")
        entries = scan_externals(self.index.root)
        report = format_catalog_markdown(entries)
        result.plan = {
            "summary": "List $externals from _description files (no LLM)",
            "mode": "externals_catalog",
            "count": len(entries),
            "files_to_read": [e.rel_path for e in entries],
            "grep_terms": [],
            "search_queries": [],
        }
        result.context_files = {e.rel_path: e.description for e in entries}
        result.diff_result = DiffResult(raw_output=report, diffs=[])
        result.report = report
        self._write_json(session_dir / "plan.json", result.plan)
        self._write_json(
            session_dir / "context_index.json",
            {"files": list(result.context_files.keys()), "grep_hits": []},
        )
        (session_dir / "output.txt").write_text(report, encoding="utf-8")
        (session_dir / "report.md").write_text(report, encoding="utf-8")
        (session_dir / "catalog.json").write_text(
            json.dumps(
                [
                    {"name": e.name, "path": e.rel_path, "summary": e.summary}
                    for e in entries
                ],
                indent=2,
                ensure_ascii=False,
            )
            + "\n",
            encoding="utf-8",
        )

    def _new_session(self) -> Path:
        root = self.config.ensure_sessions_dir()
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = root / stamp
        path.mkdir(parents=True, exist_ok=True)
        return path

    @staticmethod
    def _write_json(path: Path, data: object) -> None:
        path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    def _plan(
        self, request: str, tree: str, chat_history: ChatHistory | None = None
    ) -> dict:
        tree, _ = trim_plain_text(tree, budget_chars=8000, label="plan tree")
        prompt = (
            f"# Change request\n{request}\n\n"
            f"# Repository file tree (partial)\n{tree}\n\n"
            f"# Language note\n{LANG_REFERENCE}\n\n"
            "Output the JSON plan object."
        )
        try:
            plan = self.model.complete_json(
                prompt,
                system=PLANNER_SYSTEM,
                history=chat_history,
                max_tokens=self.config.plan_max_tokens,
            )
        except (json.JSONDecodeError, ValueError) as exc:
            self._status(f"Plan parse fallback ({exc})")
            plan = self._fallback_plan(request)
        for key in ("files_to_read", "grep_terms", "search_queries"):
            if key not in plan or not isinstance(plan[key], list):
                plan[key] = []
        if "summary" not in plan:
            plan["summary"] = "Analyze request and propose diffs"
        plan["files_to_read"] = self._filter_plan_paths(plan.get("files_to_read") or [])
        return plan

    def _enrich_plan_from_conversation(
        self, plan: dict, conversation: list[HistoryTurn]
    ) -> dict:
        if not conversation:
            return plan
        last = conversation[-1]
        paths = list(plan.get("files_to_read") or [])
        for rel in reversed(last.files):
            if rel not in paths:
                paths.insert(0, rel)
        plan["files_to_read"] = self._filter_plan_paths(paths)
        return plan

    def _filter_plan_paths(self, paths: list) -> list[str]:
        indexed = {e.rel_path for e in self.index.entries()}
        out: list[str] = []
        for raw in paths:
            rel = str(raw).strip().replace("\\", "/").lstrip("/")
            if not rel or is_blocked_path(rel):
                continue
            if rel not in indexed:
                try:
                    self.index.resolve(rel)
                except ValueError:
                    continue
            if rel not in out:
                out.append(rel)
            if len(out) >= self.config.max_context_files:
                break
        return out

    def _fallback_plan(self, request: str) -> dict:
        hits = search_files(self.index, request, limit=6)
        files = [e.rel_path for e in hits]
        for must in ("_lang_desc", "AH_CODEGEN_INSTRUCTIONS.md"):
            if must not in files and (self.index.root / must).is_file():
                files.insert(0, must)
        return {
            "summary": "Heuristic plan from keyword search",
            "files_to_read": self._filter_plan_paths(files),
            "grep_terms": [],
            "search_queries": [],
        }

    def _gather_files(
        self, plan: dict
    ) -> tuple[dict[str, str], list[tuple[str, int, str]]]:
        files: dict[str, str] = {}
        for rel in self._filter_plan_paths(plan.get("files_to_read") or []):
            self._check_cancel()
            try:
                files[rel] = self.index.read(rel)
            except (OSError, ValueError) as exc:
                files[rel] = f"[read error: {exc}]"

        grep_hits: list[tuple[str, int, str]] = []
        for term in plan.get("grep_terms") or []:
            self._check_cancel()
            term = str(term).strip()
            if not term:
                continue
            grep_hits.extend(self.index.grep(term, limit=15))
        return files, grep_hits[:40]

    def _gather_search(self, queries: list) -> list[dict[str, str]]:
        rows: list[dict[str, str]] = []
        for q in queries[:3]:
            self._check_cancel()
            q = str(q).strip()
            if not q:
                continue
            for row in web_search(q, limit=self.config.search_limit):
                rows.append({"query": q, **row})
        return rows

    def _generate(
        self,
        request: str,
        plan: dict,
        context_files: dict[str, str],
        grep_hits: list[tuple[str, int, str]],
        search_results: list[dict[str, str]],
        tree: str,
        chat_history: ChatHistory | None = None,
    ) -> str:
        parts = AgentPromptParts(
            request=request,
            plan_summary=str(plan.get("summary", "")),
            tree=tree,
            context_files=context_files,
            grep_hits=grep_hits[:30],
            search_results=search_results[:10],
        )
        return self.model.complete_agent(
            parts, system=GENERATOR_SYSTEM, history=chat_history
        )
