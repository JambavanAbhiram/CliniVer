"""The verification summary table (claim, verdict, confidence, evidence dates, links)."""

from cliniver.service.summary import md_escape, summary_csv, summary_markdown, summary_rows


def _result():
    ev = lambda src, link, date, doc="d": {"source": src, "link": link, "date": date, "title": "T", "doc_id": doc, "text": "x"}
    return {"claims": [
        {"claim": "it lowers HbA1c", "verified_text": "Metformin lowers HbA1c | by 1%", "verdict": "SUP", "confidence": 0.8625,
         "evidence": [ev("pubmed", "https://pubmed.ncbi.nlm.nih.gov/1/", "2019-02-15", "pm1"), ev("openfda_label", None, None, "fda1"),
                      ev("openfda_label", None, None, "fda1")],   # a second passage of the same label
         "evidence_dates": ["2019-02-15", None]},
        {"claim": "c2", "verdict": "INS", "confidence": 0.31, "evidence": [ev("who_gho", "https://who.int/x", "1998-12-31")],
         "evidence_dates": ["1998-12-31"], "second_opinion": {"label": "INS", "agrees": False}},
    ]}


def test_rows_use_the_verified_text_and_flag_llm_decisions():
    rows = summary_rows(_result())
    assert rows[0]["claim"] == "Metformin lowers HbA1c | by 1%" and rows[0]["dates"] == ["2019-02-15"]
    assert rows[0]["links"][0] == ("PubMed", "https://pubmed.ncbi.nlm.nih.gov/1/")
    assert rows[1]["verdict_text"].endswith("†")               # LLM disagreed and its label is the verdict
    assert not rows[0]["verdict_text"].endswith("†")
    assert len(rows[0]["links"]) == 2                           # passages of one document are listed once


def test_markdown_table_escapes_pipes_and_links_evidence():
    md = summary_markdown(summary_rows(_result()))
    lines = md.splitlines()
    assert lines[0].startswith("| # | Claim | Verdict | Confidence | Evidence dates | Evidence |")
    assert "Metformin lowers HbA1c \\| by 1%" in md                # a pipe in a claim does not break the table
    assert "[PubMed 1](https://pubmed.ncbi.nlm.nih.gov/1/)" in md and "FDA label 2" in md   # no link → plain label
    assert "86%" in md and all(l.count("|") - l.count("\\|") == 7 for l in lines)


def test_csv_and_escape():
    csv_text = summary_csv(summary_rows(_result()))
    assert csv_text.splitlines()[0].startswith("#,claim,verdict,confidence")
    assert md_escape("a*b_[c]") == "a\\*b\\_\\[c\\]"
