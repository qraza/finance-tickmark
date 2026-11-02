"""The metric catalogue: what the LLM is allowed to ask for.

Loaded from dbt's ``target/semantic_manifest.json`` (written by ``dbt parse``), so it
needs no warehouse connection and stays in lockstep with the YAML in ``dbt/models``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

DEFAULT_MANIFEST = Path(__file__).resolve().parents[2] / "dbt" / "target" / "semantic_manifest.json"

TIME_GRAINS = ("month", "quarter", "year")


@dataclass(frozen=True)
class Metric:
    name: str
    label: str
    description: str
    owner: str | None
    definition: str | None


@dataclass(frozen=True)
class Dimension:
    name: str          # fully qualified, e.g. cost_centre__cost_centre_name
    description: str | None
    values: tuple[str, ...] = ()  # optional known values, filled in by the engine


@dataclass
class Catalogue:
    metrics: dict[str, Metric]
    dimensions: dict[str, Dimension]
    time_dimensions: tuple[str, ...] = field(
        default_factory=lambda: tuple(f"metric_time__{g}" for g in TIME_GRAINS)
    )

    @classmethod
    def load(cls, manifest_path: Path = DEFAULT_MANIFEST) -> "Catalogue":
        if not manifest_path.exists():
            raise FileNotFoundError(
                f"{manifest_path} not found. Run `make build` (dbt parse) first."
            )
        m = json.loads(manifest_path.read_text())

        metrics = {}
        for mt in m["metrics"]:
            meta = (mt.get("config") or {}).get("meta") or {}
            metrics[mt["name"]] = Metric(
                name=mt["name"],
                label=mt.get("label") or mt["name"],
                description=mt.get("description") or "",
                owner=meta.get("owner"),
                definition=meta.get("definition"),
            )

        dims = {}
        for sm in m["semantic_models"]:
            primary = next((e["name"] for e in sm["entities"] if e["type"] == "primary"), None)
            if primary is None:
                continue
            for d in sm["dimensions"]:
                if d["type"] == "time":
                    continue  # time is exposed through metric_time
                qualified = f"{primary}__{d['name']}"
                dims[qualified] = Dimension(qualified, d.get("description"))
        return cls(metrics=metrics, dimensions=dims)

    @property
    def group_by_names(self) -> list[str]:
        return sorted(self.dimensions) + list(self.time_dimensions)

    def describe(self) -> str:
        """Plain-text catalogue for the LLM prompt and for the RAG index."""
        lines = ["Metrics:"]
        for mt in self.metrics.values():
            owner = f" Owner: {mt.owner}." if mt.owner else ""
            lines.append(f"- {mt.name} ({mt.label}): {mt.description}{owner}")
        lines.append("Dimensions:")
        for d in self.dimensions.values():
            desc = f": {d.description}" if d.description else ""
            vals = f" Values: {', '.join(d.values)}." if d.values else ""
            lines.append(f"- {d.name}{desc}{vals}")
        lines.append("Time: metric_time at " + ", ".join(TIME_GRAINS) + " grain.")
        return "\n".join(lines)

    def with_dimension_values(self, values: dict[str, list[str]]) -> "Catalogue":
        """Attach the known values of each dimension, used for the prompt and for validation."""
        dims = {
            name: Dimension(d.name, d.description, tuple(values.get(name, d.values)))
            for name, d in self.dimensions.items()
        }
        return Catalogue(self.metrics, dims, self.time_dimensions)
