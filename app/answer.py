"""The answer contract shared by the app, the UI and the eval runner."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field


@dataclass
class Citation:
    source: str          # corpus-relative path, e.g. commentary/2026-07.md
    section: str | None = None
    quote: str | None = None


@dataclass
class Answer:
    text: str
    numbers: dict[str, float] = field(default_factory=dict)  # name -> value, for numeric scoring
    sql: str | None = None
    citations: list[Citation] = field(default_factory=list)
    refused: bool = False
    refusal_reason: str | None = None
    semantic_query: dict | None = None  # what the LLM asked for, kept for audit and debugging
    warnings: list[str] = field(default_factory=list)  # e.g. an explanation rejected by the grounding check
    # The parts of `text`, kept separate for the UI:
    question: str | None = None
    scope: str | None = None            # "Q3 2026, cost centre name Engineering"
    table: dict | None = None           # {"columns": [...], "labels": {col: label}, "rows": [[...], ...]}
    drivers: list[dict] = field(default_factory=list)  # drill-down rows behind a "why" answer
    explanation: str | None = None      # Claude's written answer (or the fallback passage), without refs

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Answer":
        d = dict(d)
        d["citations"] = [Citation(**c) for c in d.get("citations", [])]
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})

    @classmethod
    def refusal(cls, reason: str) -> "Answer":
        return cls(text=f"I can't answer that from the governed data: {reason}",
                   refused=True, refusal_reason=reason)
