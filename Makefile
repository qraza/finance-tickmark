export DBT_PROFILES_DIR := $(CURDIR)/dbt
export FINANCE_DB_PATH := $(CURDIR)/finance.duckdb

.PHONY: setup data build test evals evals-pipeline all clean

setup:
	pip install -e ".[dev]"

data:            ## generate the synthetic ledger and commentary
	python data_gen/generate.py

build: data      ## dbt build + semantic layer validation + metric definitions for RAG
	cd dbt && dbt build --quiet && mf validate-configs --skip-dw
	python -m app.semantic.export_definitions

test:
	pytest -q

evals:           ## oracle evals: data + semantic layer + retrieval, no LLM. CI gate.
	python -m evals.run --cases evals/golden/finance.yaml --system evals.finance:oracle

evals-pipeline:  ## full system with the LLM (needs ANTHROPIC_API_KEY)
	python -m evals.run --cases evals/golden/finance.yaml --system evals.finance:pipeline

all: build test evals

clean:
	rm -rf finance.duckdb dbt/target dbt/logs evals/results
