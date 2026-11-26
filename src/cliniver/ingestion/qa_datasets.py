"""Medical long-form QA datasets used as query sources and extraction references (NB03).

* **K-QA** (Manes et al., 2024; MIT): 201 real patient questions with physician-written answers and
  physician-written *Must_have* / *Nice_to_have* statements → human reference statements.
* **MedLFQA** (Jeong et al., 2024, OLAPH; Hugging Face card: CC-BY-4.0): K-QA golden/silver, LiveQA, MedicationQA,
  HealthSearchQA re-packaged with *Must_have* / *Nice_to_have* statements. For the non-K-QA-golden subsets the
  statements were produced with an LLM by the dataset authors, so they are **not** human labels. Each component
  dataset keeps its own original terms.
"""

from __future__ import annotations

import ast
import json
import re
from dataclasses import dataclass, field


@dataclass(frozen=True)
class QAItem:
    qid: str
    source: str  # kqa_golden | kqa_silver | live_qa | medication_qa | healthsearch_qa
    question: str
    answer: str
    must_have: tuple[str, ...]
    nice_to_have: tuple[str, ...]
    human_statements: bool
    meta: dict = field(default_factory=dict, compare=False, hash=False)


def _statements(v) -> tuple[str, ...]:
    if isinstance(v, list):
        return tuple(str(x).strip() for x in v if str(x).strip())
    if not v:
        return ()
    try:
        x = ast.literal_eval(v)  # stored as a Python-literal list string
        return tuple(str(s).strip() for s in x if str(s).strip())
    except (ValueError, SyntaxError):
        return (str(v).strip(),)


def parse_jsonl(raw: bytes, source: str, *, human: bool) -> list[QAItem]:
    out = []
    for i, line in enumerate(raw.decode("utf-8").splitlines()):
        if not line.strip():
            continue
        r = json.loads(line)
        out.append(QAItem(f"{source}:{i}", source, " ".join(str(r["Question"]).split()), str(r.get("Free_form_answer") or "").strip(),
                          _statements(r.get("Must_have")), _statements(r.get("Nice_to_have")), human,
                          {k: r[k] for k in ("Sources", "ICD_10_diag") if k in r}))
    return out


def domain_regex(drug_names: set[str]) -> re.Pattern:
    """In-scope drugs (RxNorm names and synonyms) or one of the five condition areas."""
    cond = (r"diabet|hypertens|blood pressure|cholesterol|statin|lipid|triglycerid|atrial fibrillation|\bafib\b|"
            r"anticoagul|blood thinner|heart failure|\ba1c\b|insulin|blood sugar|glucose")
    names = "|".join(re.escape(n) for n in sorted(drug_names, key=len, reverse=True))
    return re.compile(rf"\b(?:{names})\b|{cond}", re.I)
