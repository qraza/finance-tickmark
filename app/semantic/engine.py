"""Compile a SemanticQuery with MetricFlow, run it on DuckDB, and keep the SQL."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path

from .catalogue import Catalogue
from .query import SemanticQuery

REPO = Path(__file__).resolve().parents[2]
DBT_DIR = REPO / "dbt"


@dataclass
class QueryResult:
    query: SemanticQuery
    columns: list[str]
    rows: list[tuple]
    sql: str

    def as_records(self) -> list[dict]:
        return [dict(zip(self.columns, r)) for r in self.rows]


class SemanticEngine:
    def __init__(self, dbt_dir: Path = DBT_DIR):
        self.dbt_dir = dbt_dir
        os.environ.setdefault("FINANCE_TICKMARK_DB_PATH", str(REPO / "finance-tickmark.duckdb"))
        self.catalogue = Catalogue.load(dbt_dir / "target" / "semantic_manifest.json")

    @cached_property
    def _mf(self):
        # Imported lazily: MetricFlow is heavy and unit tests don't need it.
        from dbt_metricflow.cli.cli_configuration import CLIConfiguration

        logging.getLogger("metricflow").setLevel(logging.WARNING)
        cfg = CLIConfiguration()
        cfg.setup(dbt_profiles_path=self.dbt_dir, dbt_project_path=self.dbt_dir,
                  configure_file_logging=False)
        return cfg.mf

    def load_dimension_values(self) -> Catalogue:
        """Fill in each dimension's known values from the warehouse (once per engine).

        The values go into the LLM prompt, so filters use exact names ("Engineering"),
        and into validation, so a misspelt or invented value is refused, not silently empty.
        """
        if any(d.values for d in self.catalogue.dimensions.values()):
            return self.catalogue
        # Each simple metric is filtered to one account type, so take the union across all three.
        base = ["revenue", "cogs", "opex"]
        values = {}
        for name in self.catalogue.dimensions:
            seen = set()
            for m in base:
                seen.update(self._mf.get_dimension_values(metric_names=[m], get_group_by_values=name))
            values[name] = sorted(seen)
        self.catalogue = self.catalogue.with_dimension_values(values)
        return self.catalogue

    def run(self, query: SemanticQuery) -> QueryResult:
        from metricflow.engine.metricflow_engine import MetricFlowQueryRequest

        query.validate(self.catalogue)
        request = MetricFlowQueryRequest.create(**query.to_mf_args())
        sql = self._mf.explain(mf_request=request).sql_statement.without_descriptions.sql
        table = self._mf.query(mf_request=request).result_df
        return QueryResult(query=query, columns=list(table.column_names),
                           rows=[tuple(r) for r in table.rows], sql=sql)
