"""Ask a question from the terminal.

    python -m app.ask "What was Q3 opex against budget?"
    python -m app.ask --sql "Which cost centre overspent most in Q3?"
"""

from __future__ import annotations

import argparse
import json
import os

from .pipeline import answer


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description="Ask finance-tickmark a question.")
    p.add_argument("question")
    p.add_argument("--sql", action="store_true", help="show the SQL MetricFlow generated")
    p.add_argument("--query", action="store_true", help="show the semantic query the LLM chose")
    args = p.parse_args(argv)

    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise SystemExit("ANTHROPIC_API_KEY not set: add it to .env (see .env.example).")
    ans = answer(args.question)
    print(ans.text)
    for w in ans.warnings:
        print(f"\nNote: {w}")
    if args.query and ans.semantic_query:
        print("\nSemantic query:\n" + json.dumps(ans.semantic_query, indent=2))
    if args.sql and ans.sql:
        print("\nSQL:\n" + ans.sql)


if __name__ == "__main__":
    main()
