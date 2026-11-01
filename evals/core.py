"""Generic eval core: cases, scorers, aggregation. Nothing here knows about finance.

A *system* is any callable ``(case) -> output``. A *scorer* is any callable
``(case, output) -> float | None`` (None = not applicable to this case).
Phase 2 reuses this module unchanged for dbt-sentinel with its own cases and scorers.
"""

from __future__ import annotations

import importlib
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import yaml


@dataclass
class Case:
    id: str
    question: str
    expected: dict[str, Any] = field(default_factory=dict)
    tags: list[str] = field(default_factory=list)
    oracle: dict[str, Any] | None = None   # system-specific hints, e.g. a reference query


Scorer = Callable[[Case, Any], "float | None"]


@dataclass
class CaseResult:
    case_id: str
    scores: dict[str, float | None]
    latency_s: float
    error: str | None = None
    output: Any = None


def load_cases(path: Path) -> list[Case]:
    raw = yaml.safe_load(path.read_text())
    return [Case(**c) for c in raw["cases"]]


def load_callable(spec: str) -> Callable:
    """'package.module:attr' -> object."""
    mod, _, attr = spec.partition(":")
    return getattr(importlib.import_module(mod), attr)


def run(cases: list[Case], system: Callable, scorers: dict[str, Scorer]) -> list[CaseResult]:
    results = []
    for case in cases:
        t0 = time.perf_counter()
        try:
            out = system(case)
            err = None
        except Exception:  # a crash scores zero, it doesn't stop the run
            out, err = None, traceback.format_exc(limit=3)
        latency = time.perf_counter() - t0
        if err:
            scores = {name: 0.0 for name in scorers}
        else:
            scores = {name: fn(case, out) for name, fn in scorers.items()}
        results.append(CaseResult(case.id, scores, latency, err, out))
    return results


def summarise(results: list[CaseResult]) -> dict[str, dict[str, float]]:
    names = {n for r in results for n in r.scores}
    summary = {}
    for n in sorted(names):
        vals = [r.scores[n] for r in results if r.scores.get(n) is not None]
        summary[n] = {"mean": sum(vals) / len(vals) if vals else float("nan"), "n": len(vals)}
    return summary
