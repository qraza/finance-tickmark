export DBT_PROFILES_DIR := $(CURDIR)/dbt
export FINANCE_TICKMARK_DB_PATH := $(CURDIR)/finance-tickmark.duckdb

.PHONY: setup data build test evals evals-pipeline ask all clean

setup:
	pip install -e ".[dev,llm]"

data:            ## generate the synthetic ledger and commentary
	python data_gen/generate.py

build: data      ## dbt build + semantic layer validation + metric definitions for RAG
	cd dbt && dbt build --quiet && mf validate-configs --skip-dw
	python -m app.semantic.export_definitions

test:
	pytest -q

evals:           ## oracle evals: data + semantic layer + retrieval, no LLM. CI gate.
	python -m evals.run --cases evals/golden/finance.yaml --system evals.finance:oracle

evals-pipeline:  ## full system with the LLM (needs ANTHROPIC_API_KEY in .env)
	@python -c "import app, os, sys; sys.exit(0 if os.environ.get('ANTHROPIC_API_KEY') else 'ANTHROPIC_API_KEY not set: add it to .env')"
	python -m evals.run --cases evals/golden/finance.yaml --system evals.finance:pipeline --out evals/results/pipeline --workers 6

ask:             ## make ask Q="What was Q3 opex against budget?"
	python -m app.ask --sql "$(Q)"

all: build test evals

clean:
	rm -rf finance-tickmark.duckdb dbt/target dbt/logs evals/results
