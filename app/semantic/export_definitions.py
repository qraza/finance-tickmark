"""Write the metric catalogue to corpus/metrics/metric_definitions.md so RAG can cite it.

Generated from the semantic manifest, so a definition can never drift from the YAML.
Run by `make build` after `dbt parse`.
"""

from pathlib import Path

from .catalogue import Catalogue

OUT = Path(__file__).resolve().parents[2] / "corpus" / "metrics" / "metric_definitions.md"


def main() -> None:
    cat = Catalogue.load()
    lines = [
        "# Metric definitions",
        "",
        "Generated from the dbt semantic layer. Do not edit by hand; edit dbt/models/marts/metrics.yml.",
        "",
    ]
    for m in cat.metrics.values():
        lines += [f"## {m.label} (`{m.name}`)", "", m.description, ""]
        if m.definition:
            lines += [f"Definition: {m.definition}", ""]
        if m.owner:
            lines += [f"Owner: {m.owner}", ""]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(lines))
    print(f"wrote {len(cat.metrics)} definitions to {OUT}")


if __name__ == "__main__":
    main()
