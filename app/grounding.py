"""Check that every £ amount and percentage Claude writes appears in what it was given.

Claude writes the explanation for "why" and document answers, so it could in principle
state a number nobody gave it. This check parses each money amount (£138k, £5.63m,
£1,054k, -£49k) and each percentage in the text, and requires it to match a figure in
the semantic layer's result or a quoted passage, within rounding. Anything else fails
the answer, and the pipeline falls back to the figures and passages without prose.

Counts and years ("four roles", "2026") are not checked: they aren't financial figures.
"""

from __future__ import annotations

import re

_MONEY = re.compile(r"(-)?£\s?(-)?(\d[\d,]*(?:\.\d+)?)\s?([km])?\b", re.IGNORECASE)
_PCT = re.compile(r"(-?\d+(?:\.\d+)?)\s?%")
_SCALE = {"k": 1e3, "m": 1e6, None: 1.0, "": 1.0}


def money_values(text: str) -> list[float]:
    out = []
    for neg1, neg2, num, unit in _MONEY.findall(text):
        v = float(num.replace(",", "")) * _SCALE[(unit or "").lower() or None]
        out.append(-v if (neg1 or neg2) else v)
    return out


def pct_values(text: str) -> list[float]:
    return [float(p) for p in _PCT.findall(text)]


def _close(v: float, allowed: list[float], rel: float) -> bool:
    # Signs are compared loosely: "£264k over" and "a variance of £264k" are the same figure.
    return any(abs(abs(v) - abs(a)) <= rel * max(abs(a), 1.0) for a in allowed)


def ungrounded(text: str, sources: list[str], raw_numbers: list[float] = ()) -> list[str]:
    """Figures in `text` that match nothing in `sources` or `raw_numbers`. Empty = grounded."""
    allowed_money = [v for s in sources for v in money_values(s)] + list(raw_numbers)
    allowed_pct = [v for s in sources for v in pct_values(s)]
    bad = []
    for m in _MONEY.finditer(text):
        v = money_values(m.group(0))[0]
        # 1.5% tolerance covers "£5.6m" quoted from "£5.63m"; a wrong figure is far outside it.
        if not _close(v, allowed_money, 0.015):
            bad.append(m.group(0).strip())
    for m in _PCT.finditer(text):
        if not _close(float(m.group(1)), allowed_pct, 0.02):
            bad.append(m.group(0).strip())
    return bad
