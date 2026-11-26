"""COVID-QA (Möller et al., 2020): expert-annotated extractive QA over full-text CORD-19 articles.

Used in NB02 ONLY as a span-level retrieval benchmark for chunking. It is not a source of clinical
evidence for CliniVer (different domain). Repository license: Apache-2.0; the articles keep their own
licenses (stated inside each context). Nothing is redistributed: the raw file is archived locally.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

from cliniver.evaluation.span_retrieval import Gold
from cliniver.ingestion.http import RateLimitedClient, RawStore, cached_get


@dataclass(frozen=True)
class QADoc:
    doc_id: str
    text: str


@dataclass(frozen=True)
class Question:
    qid: str
    text: str
    gold: Gold


def fetch(url: str, store: RawStore) -> bytes:
    client = RateLimitedClient("github_raw", min_interval_s=1.0, max_in_flight=1)
    return cached_get(client, store, "COVID-QA", "json", url)


def parse(raw: bytes) -> tuple[list[QADoc], list[Question]]:
    data = json.loads(raw)["data"]
    docs, qs = [], []
    for art in data:
        for para in art["paragraphs"]:
            did = f"covidqa:{para['document_id']}"
            ctx = para["context"]
            docs.append(QADoc(did, ctx))
            for q in para["qas"]:
                if q.get("is_impossible") or not q["answers"]:
                    continue
                a = q["answers"][0]
                s, e = a["answer_start"], a["answer_start"] + len(a["text"])
                if ctx[s:e] != a["text"]:
                    raise ValueError(f"answer offset mismatch in question {q['id']}")
                qs.append(Question(str(q["id"]), q["question"], Gold(str(q["id"]), did, s, e)))
    return docs, qs
