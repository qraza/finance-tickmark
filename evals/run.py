"""CLI: run a golden set against a system and gate on thresholds.

    python -m evals.run --cases evals/golden/finance.yaml --system evals.finance:oracle

Exits 1 if any metric's mean is below its threshold, so CI fails the build.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import sys
from pathlib import Path

from evals.core import load_callable, load_cases, run, summarise

DEFAULT_THRESHOLDS = {"numeric_accuracy": 0.95, "citation_recall": 0.8, "refusal_correctness": 1.0}


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--cases", type=Path, required=True)
    p.add_argument("--system", required=True, help="module:callable taking a Case")
    p.add_argument("--scorers", default="evals.finance:SCORERS", help="module:dict of scorers")
    p.add_argument("--tag", help="only run cases with this tag")
    p.add_argument("--out", type=Path, default=Path("evals/results"))
    p.add_argument("--threshold", action="append", default=[], help="metric=value, overrides defaults")
    p.add_argument("--workers", type=int, default=1, help="cases to run at once (LLM systems: 4-8)")
    args = p.parse_args(argv)

    cases = load_cases(args.cases)
    if args.tag:
        cases = [c for c in cases if args.tag in c.tags]
    thresholds = dict(DEFAULT_THRESHOLDS)
    for t in args.threshold:
        k, v = t.split("=")
        thresholds[k] = float(v)

    def progress(done: int, total: int, r) -> None:
        # ✗ only for metrics with a threshold; precision is reported, not gated
        ok = not r.error and all(v is None or v >= 1 for k, v in r.scores.items() if k in thresholds)
        print(f"[{done:>{len(str(total))}}/{total}] {'✓' if ok else '✗'} {r.case_id} ({r.latency_s:.1f}s)",
              file=sys.stderr, flush=True)

    results = run(cases, load_callable(args.system), load_callable(args.scorers),
                  workers=args.workers, progress=progress)
    summary = summarise(results)


    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "results.json").write_text(json.dumps(
        {"system": args.system, "summary": summary,
         "cases": [{"id": r.case_id, "scores": r.scores, "latency_s": round(r.latency_s, 3), "error": r.error,
                    "output": _jsonable(r.output)}
                   for r in results]}, indent=2, default=str))

    lines = ["| metric | mean | n | threshold | |", "|---|---|---|---|---|"]
    failed = False
    for name, s in summary.items():
        thr = thresholds.get(name)
        ok = thr is None or (not math.isnan(s["mean"]) and s["mean"] >= thr) or s["n"] == 0
        failed |= not ok
        lines.append(f"| {name} | {s['mean']:.3f} | {s['n']} | {thr if thr is not None else '-'} | {'✅' if ok else '❌'} |")
    table = "\n".join(lines)
    (args.out / "summary.md").write_text(table + "\n")
    print(table)

    for r in results:
        bad = {k: round(v, 2) for k, v in r.scores.items() if v is not None and v < 1}
        if bad or r.error:
            print(f"\n{r.case_id}: {bad}" + (f"\n{r.error}" if r.error else ""))
            for attr in ("numbers", "refusal_reason"):
                val = getattr(r.output, attr, None)
                if val:
                    print(f"  got {attr}: {val}")
            cited = [c.source for c in getattr(r.output, "citations", []) or []]
            if cited:
                print(f"  got citations: {cited}")
    print(f"\nPer-case outputs: {args.out / 'results.json'}")
    return 1 if failed else 0


def _jsonable(out):
    if out is None:
        return None
    if dataclasses.is_dataclass(out):
        return dataclasses.asdict(out)
    return repr(out)


if __name__ == "__main__":
    sys.exit(main())
