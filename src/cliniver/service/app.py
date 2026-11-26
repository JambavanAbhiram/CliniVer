"""CliniVer Streamlit interface (local research demo; localhost only).

    .venv\\Scripts\\python.exe scripts\\run_app.py

Tabs:
1. Ask a question: the local LLM answers from retrieved evidence, then the answer is verified claim by claim.
2. Check another answer: verify an answer written by any other AI.
(Expert review and evaluation results were removed from the interface on the owner's request, 2026-11-23; the review
store `cliniver.service.review` and the experiment records remain.)

Research prototype: never medical advice.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from cliniver.service.engine import Engine
from cliniver.service.summary import VERDICT_LABEL, summary_csv, summary_markdown, summary_rows

st.set_page_config(page_title="CliniVer", page_icon="🩺", layout="wide")


@st.cache_resource(show_spinner="Loading the corpus and models (about 30 s, once)…")
def get_engine() -> Engine:
    e = Engine(log=lambda m: print(f"[engine] {m}", flush=True))
    e.load()
    return e


st.markdown("<h1 style='font-size:3rem;margin:0 0 .1rem;color:#0f6c7b;letter-spacing:-.01em'>CliniVer</h1>", unsafe_allow_html=True)
engine = get_engine()


def show_result(res: dict, key: str) -> None:
    t = res.get("timings", {})
    st.caption(" · ".join(f"{k.replace('_s', '').replace('_', ' ')} {v:.1f} s" for k, v in t.items()) + f" · total {sum(t.values()):.1f} s")
    for n in res.get("notes", []):
        st.info(n)
    rows = summary_rows(res)
    if not rows:
        st.warning("No checkable claim was found.")
        return
    counts = pd.Series([r["verdict"] for r in rows]).value_counts()
    cols = st.columns(len(counts))
    for c, (v, n) in zip(cols, counts.items()):
        c.metric(VERDICT_LABEL.get(v, v), int(n))
    st.markdown("#### Verification summary")
    st.markdown(summary_markdown(rows))
    st.caption("Confidence = calibrated probability that the verifier's label is correct. † verdict decided by the LLM second opinion "
               "(low-confidence claim). Evidence 1 is the decisive passage.")
    st.download_button("Download the table (CSV)", summary_csv(rows), file_name="cliniver_verification.csv", mime="text/csv", key=f"dl_{key}")
    with st.expander("Claim details: probabilities, rules, second opinion and evidence text"):
        for r, c in zip(rows, res["claims"]):
            st.markdown(f"**{r['n']}. {c['claim']}**")
            if c.get("verified_text") and c["verified_text"] != c["claim"]:
                st.caption(f"Verified as the standalone rewrite: {c['verified_text']}")
            st.caption(f"Verifier label {c['verifier_label']} · probabilities "
                       + ", ".join(f"{k} {v:.2f}" for k, v in c["probabilities"].items())
                       + (f" · rules: {'; '.join(c['rules_applied'])}" if c.get("rules_applied") else ""))
            if c.get("second_opinion"):
                so = c["second_opinion"]
                st.caption(f"Second opinion ({so['model']}): {so['label']} ({'agrees' if so['agrees'] else 'disagrees'}). {so['rationale']}")
            for k, e in enumerate(c["evidence"], 1):
                st.markdown(f"[{k}] *{e['source']} · {e.get('date') or 'undated'}* · " + (f"[{e['title']}]({e['link']})" if e.get("link") else e["title"]))
                st.text(e["text"][:1200])
            st.divider()


tab_ask, tab_check = st.tabs(["Ask a question", "Check another answer"])

with tab_ask:
    st.markdown("Ask a question about adult cardiometabolic drug therapy. CliniVer retrieves evidence, writes an answer with a local "
                "open-weight model, then **verifies every claim of that answer** against the corpus.")
    q = st.text_area("Question", key="ask_q", height=80, placeholder="e.g. What are the main risks of metformin, and when should it not be used?")
    go = st.button("Generate and verify", type="primary", key="ask_go")
    if go and not q.strip():
        st.warning("Type a question first.")
    elif go:
        with st.spinner("Retrieving evidence, generating the answer, extracting and verifying claims (typically 30–60 s)…"):
            try:
                st.session_state["ask_res"] = engine.ask(q)
            except (ValueError, RuntimeError) as exc:
                st.error(str(exc))
    res = st.session_state.get("ask_res")
    if res:
        g = res.get("generated", {})
        st.markdown("#### Generated answer")
        st.caption(f"Written by {g.get('model')} (strategy {g.get('strategy')}) from the {len(g.get('passages', []))} passages below. "
                   "The answer itself is not trusted: the summary table verifies each of its claims.")
        st.info(g.get("answer") or res.get("answer") or "(no answer)")
        with st.expander("Evidence passages given to the model (and the search queries that found them)"):
            st.caption("Search queries: " + " · ".join(g.get("search_queries", [])))
            for k, e in enumerate(g.get("passages", []), 1):
                st.markdown(f"**[{k}]** *{e['source']} · {e.get('date') or 'undated'}* · " + (f"[{e['title']}]({e['link']})" if e.get("link") else e["title"]))
                st.text(e["text"][:1000])
        show_result(res, "ask")

with tab_check:
    st.markdown("Paste an answer written by another AI system to audit it claim by claim.")
    q2 = st.text_area("Question", key="chk_q", height=70)
    a2 = st.text_area("Answer to check", key="chk_a", height=180)
    go2 = st.button("Verify", key="chk_go")
    if go2 and not a2.strip():
        st.warning("Paste an answer first.")
    elif go2:
        with st.spinner("Extracting and verifying claims…"):
            try:
                st.session_state["chk_res"] = engine.verify(q2, a2)
            except (ValueError, RuntimeError) as exc:
                st.error(str(exc))
    if st.session_state.get("chk_res"):
        show_result(st.session_state["chk_res"], "chk")

st.divider()
st.caption(engine.status()["disclaimer"])
