"""Persistent conversation history — queries and outputs."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from brain.config import BrainConfig, load_config

PREVIEW_CHARS = 4000


@dataclass
class HistoryTurn:
    tm: str
    request: str
    report: str
    plan_summary: str = ""
    mode: str = ""
    session: str | None = None
    error: str | None = None
    elapsed_s: float = 0.0
    files: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        preview = self.report
        if len(preview) > PREVIEW_CHARS:
            preview = preview[:PREVIEW_CHARS] + "\n\n[... truncated in saves/conversation.json ...]"
        return {
            "tm": self.tm,
            "request": self.request,
            "report": preview,
            "plan_summary": self.plan_summary,
            "mode": self.mode,
            "session": self.session,
            "error": self.error,
            "elapsed_s": self.elapsed_s,
            "files": self.files,
        }

    @classmethod
    def from_dict(cls, data: dict) -> HistoryTurn:
        return cls(
            tm=str(data.get("tm", "")),
            request=str(data.get("request", "")),
            report=str(data.get("report", "")),
            plan_summary=str(data.get("plan_summary", "")),
            mode=str(data.get("mode", "")),
            session=data.get("session"),
            error=data.get("error"),
            elapsed_s=float(data.get("elapsed_s", 0)),
            files=list(data.get("files") or []),
        )


class ConversationStore:
    def __init__(self, config: BrainConfig | None = None):
        self.config = config or load_config()
        self.path = self.config.brain_dir / "saves" / "conversation.json"
        self._turns: list[HistoryTurn] = []

    def load(self) -> None:
        if self.path.is_file():
            try:
                data = json.loads(self.path.read_text(encoding="utf-8"))
                if isinstance(data, list):
                    self._turns = [
                        HistoryTurn.from_dict(item)
                        for item in data
                        if isinstance(item, dict)
                    ]
                    return
            except (json.JSONDecodeError, OSError):
                pass
        self._turns = []
        self._migrate_requests_json()

    def _migrate_requests_json(self) -> None:
        legacy = self.config.brain_dir / "saves" / "requests.json"
        if not legacy.is_file():
            return
        try:
            data = json.loads(legacy.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return
        if not isinstance(data, list):
            return
        for item in data:
            if not isinstance(item, dict):
                continue
            self._turns.append(
                HistoryTurn(
                    tm=str(item.get("tm", "")),
                    request=str(item.get("request", "")),
                    report="",
                    plan_summary=str(item.get("plan_summary", "")),
                    session=item.get("session"),
                    error=item.get("error"),
                    elapsed_s=float(item.get("elapsed_s", 0)),
                    files=list(item.get("files") or []),
                )
            )
        if self._turns:
            self.save()

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(
            json.dumps([t.to_dict() for t in self._turns], indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    def append(self, turn: HistoryTurn, *, max_turns: int = 50) -> None:
        self._turns.append(turn)
        if len(self._turns) > max_turns:
            self._turns = self._turns[-max_turns:]
        self.save()

    def turns(self) -> list[HistoryTurn]:
        return list(self._turns)

    def clear(self) -> None:
        self._turns = []
        self.save()

    def recent_for_model(self, limit: int = 8) -> list[HistoryTurn]:
        return self._turns[-limit:] if limit > 0 else []

    @staticmethod
    def to_chat_pairs(turns: list[HistoryTurn]) -> list[tuple[str, str]]:
        pairs: list[tuple[str, str]] = []
        for turn in turns:
            user = turn.request.strip()
            assistant = (turn.report or turn.error or "").strip()
            if user:
                pairs.append((user, assistant))
        return pairs

    def format_for_prompt(self, turns: list[HistoryTurn], *, max_chars: int = 12_000) -> str:
        if not turns:
            return ""
        blocks: list[str] = []
        for turn in turns:
            block = (
                f"### User ({turn.tm})\n{turn.request.strip()}\n\n"
                f"### Assistant\n{(turn.report or turn.error or '(no output)').strip()}\n"
            )
            blocks.append(block)
        text = "\n---\n".join(blocks)
        if len(text) <= max_chars:
            return text
        return text[-max_chars:]

    @staticmethod
    def turn_from_result(result: object) -> HistoryTurn:
        plan = getattr(result, "plan", {}) or {}
        return HistoryTurn(
            tm=str(datetime.now()),
            request=str(getattr(result, "request", "")),
            report=str(getattr(result, "report", "") or ""),
            plan_summary=str(plan.get("summary", "")),
            mode=str(plan.get("mode", "")),
            session=str(result.session_dir) if getattr(result, "session_dir", None) else None,
            error=getattr(result, "error", None),
            elapsed_s=float(getattr(result, "elapsed_s", 0)),
            files=list(getattr(result, "context_files", {}).keys()),
        )
