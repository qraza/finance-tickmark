"""finance-tickmark demo: ask a finance question, see the governed answer and how it was made.

    make demo          # streamlit run ui/app.py

Live mode (ANTHROPIC_API_KEY in .env and the DuckDB built): any question, answered by Claude.
Replay mode (no key, e.g. the hosted demo): the example questions, answered from
demo/answers.json, which `make demo-cache` records from a real run.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from app.answer import Answer  # noqa: E402  (loads .env too)
from app.demo import EXAMPLES, load_cache  # noqa: E402
from app.format import source_label, value  # noqa: E402

# Diverging pair, validated with the dataviz skill's checker: light steps on a light theme, dark steps on dark.
COLOURS = {"light": ("#e34948", "#2a78d6"), "dark": ("#e66767", "#3987e5")}
DB = Path(os.environ.get("FINANCE_TICKMARK_DB_PATH", REPO / "finance-tickmark.duckdb"))
LIVE = bool(os.environ.get("ANTHROPIC_API_KEY")) and DB.exists()

st.set_page_config(page_title="finance-tickmark", page_icon="✔️", layout="wide")


# --- data -------------------------------------------------------------------------

@st.cache_resource(show_spinner="Starting the semantic layer…")
def _engine():
    from app.pipeline import default_engine

    return default_engine()


@st.cache_data(show_spinner=False)
def _cache():
    return load_cache()


def ask_live(question: str) -> Answer:
    from app.pipeline import answer

    with st.spinner("Claude is planning the query, MetricFlow is running it…"):
        return answer(question, engine=_engine())


def recorded(question: str) -> Answer | None:
    answers, _ = _cache()
    return answers.get(question)


def theme() -> str:
    try:
        return st.context.theme.type or "light"
    except Exception:  # noqa: BLE001 - older Streamlit, or no browser context
        return "light"


# --- rendering --------------------------------------------------------------------

def money_or_pct(col: str, v):
    return value(col, v) if isinstance(v, (int, float)) else v


def show_figures(a: Answer) -> None:
    t = a.table
    if not t:
        return
    metric_cols = [c for i, c in enumerate(t["columns"]) if isinstance(t["rows"][0][i], (int, float))]
    st.caption(a.scope or "")
    if len(t["rows"]) == 1:
        row = dict(zip(t["columns"], t["rows"][0]))
        cols = st.columns(min(len(metric_cols), 4) or 1)
        for i, c in enumerate(metric_cols):
            cols[i % len(cols)].metric(t["labels"].get(c, c), money_or_pct(c, row[c]))
    else:
        df = pd.DataFrame(t["rows"], columns=[t["labels"].get(c, c) for c in t["columns"]])
        for c in t["columns"]:
            if c in metric_cols:
                label = t["labels"].get(c, c)
                df[label] = [money_or_pct(c, v) for v in df[label]]
        st.dataframe(df, hide_index=True, width="stretch")


def show_drivers(a: Answer) -> None:
    ds = [d for d in a.drivers if d.get("cost_centre")]
    if not ds:
        return
    st.markdown("**Largest drivers** · from the semantic layer, by cost centre and account")
    df = pd.DataFrame([{
        "driver": f"{d['cost_centre']} / {d['account']} ({d['type']}) · {value(d['metric'], d['value'])}",
        "value": d["value"],
        "effect": "Adverse" if "adverse" in value(d["metric"], d["value"]) else "Favourable",
    } for d in ds])
    # The value is in the row label, so the meaning never rests on colour alone.
    bars = alt.Chart(df).mark_bar(size=20, cornerRadiusEnd=4).encode(
        y=alt.Y("driver:N", sort=list(df["driver"]), title=None, axis=alt.Axis(labelLimit=420, labelFontSize=13)),
        x=alt.X("value:Q", title="Variance (£)", axis=alt.Axis(format="~s", gridOpacity=0.3)),
        color=alt.Color("effect:N", scale=alt.Scale(domain=["Adverse", "Favourable"], range=list(COLOURS.get(theme(), COLOURS["light"]))),
                        legend=alt.Legend(title=None, orient="top")),
        tooltip=[alt.Tooltip("driver:N", title="Driver")],
    )
    # Height per row (Step), not a fixed height: Streamlit collapses a fixed-height chart with a top legend.
    st.altair_chart(bars.properties(height=alt.Step(38)), width="stretch")
    fx = next((d for d in a.drivers if d.get("account") == "FX translation"), None)
    if fx:
        st.caption(f"Of which FX translation: {value(fx['metric'], fx['value'])} "
                   "(actuals at actual exchange rates vs the 0.79 budget rate)")


def show_explanation(a: Answer) -> None:
    if not a.explanation:
        return
    st.markdown("**Explanation** · written by Claude from the cited passages")
    st.markdown(a.explanation)
    if a.citations:
        with st.expander(f"Sources ({len(a.citations)})", expanded=False):
            for c in a.citations:
                st.markdown(f"**{source_label(c.source)}** › {c.section}  \n`{c.source}`")
                if c.quote:
                    st.markdown(f"> {c.quote}")


def show_answer(a: Answer) -> None:
    if a.refused:
        st.error(f"**Refused.** {a.refusal_reason[0].upper() + a.refusal_reason[1:]}.", icon="🛑")
        st.caption("Refusing is the designed behaviour: the system answers only from governed metrics and "
                   "its documents, and never guesses.")
    else:
        show_figures(a)
        show_drivers(a)
        show_explanation(a)
        for w in a.warnings:
            st.info(w[0].upper() + w[1:], icon="ℹ️")
    c1, c2 = st.columns(2)
    if a.semantic_query:
        with c1.expander("What Claude asked for (semantic query)"):
            st.code(json.dumps(a.semantic_query, indent=2), language="json")
    if a.sql:
        with c2.expander("SQL that ran (generated by MetricFlow, not Claude)"):
            st.code(a.sql, language="sql")


# --- page -------------------------------------------------------------------------

st.title("finance-tickmark")
st.markdown("Ask a question about **Larkspur Ltd**, a fictional UK SaaS company. Numbers come from a governed "
            "dbt semantic layer, explanations cite their documents, and every answer type is scored by evals in CI.")

with st.sidebar:
    st.subheader("How an answer is made")
    st.markdown(
        "1. **Claude plans**: it picks governed metrics, filters and a period, searches documents, or refuses.\n"
        "2. **Code checks the plan**: the period and any named cost centre must match the question.\n"
        "3. **MetricFlow writes the SQL** and DuckDB runs it. Figures are formatted in code.\n"
        "4. For *why* questions, a **drill-down** finds the drivers and each one gets its own document search.\n"
        "5. **Claude writes the explanation** from those passages; every £ and % must appear in its sources.")
    st.divider()
    _, meta = _cache()
    if LIVE:
        st.success("Live: answering with Claude", icon="🟢")
    elif meta:
        st.info(f"Replay: real answers recorded {meta.get('recorded_at', '')} with {meta.get('model', '')}"
                + (f" at commit {meta['commit']}" if meta.get("commit") else "")
                + ". Run locally with an API key to ask anything.", icon="⏺️")
    else:
        st.warning("No API key and no recorded answers. Add ANTHROPIC_API_KEY to .env, or run `make demo-cache`.")
    if meta.get("evals"):
        st.subheader("Latest eval run")
        ev = pd.DataFrame(meta["evals"]).rename(columns={"questions": "n"})
        st.dataframe(ev, hide_index=True, width="stretch")
        st.caption("Scored against a 32-question golden set; see the README for every run.")
    st.caption("All data is synthetic. Larkspur Ltd and everyone named in its documents are fictional.")

picked = st.pills("Try an example", EXAMPLES, selection_mode="single", key="example")
typed = st.text_input("Or ask your own question", placeholder="e.g. What drove the Customer Operations underspend "
                      "in August?", disabled=not LIVE,
                      help=None if LIVE else "Your own questions need live mode: an API key in .env.")
question = typed.strip() if typed.strip() else picked

if question:
    st.subheader(question)
    answer_key = f"live::{question}"
    ans = st.session_state.get(answer_key)
    if ans is None and not typed.strip():
        ans = recorded(question)  # examples replay their recorded answer: instant, and free
        if ans is not None:
            _, meta = _cache()
            st.caption(f"Recorded answer ({meta.get('recorded_at', '')}, {meta.get('model', '')}).")
            if LIVE and st.button("Re-ask live", help="Ask Claude now instead of showing the recorded answer"):
                ans = st.session_state[answer_key] = ask_live(question)
    if ans is None and LIVE:
        ans = st.session_state[answer_key] = ask_live(question)
    if ans is None:
        st.warning("That example hasn't been recorded yet. Run `make demo-cache` with an API key.")
    else:
        show_answer(ans)
