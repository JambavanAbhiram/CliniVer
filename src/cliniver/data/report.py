"""Corpus-report computations shared by NB01 Stage A and Stage B (one implementation, two corpora)."""

from __future__ import annotations

from collections import Counter
from typing import Iterable

import pandas as pd

from cliniver.data.quality import _TOKEN
from cliniver.schemas.document import Document, EntityAnnotation, RelationAnnotation

# Evidence strata from MEDLINE PublicationType, ordered from highest to lowest design tier.
# A record is assigned to the FIRST matching stratum (so an RCT that is also tagged "Review" counts as RCT).
# Members mirror the population count queries in configs/data.yaml (publication_type_strata), including the
# narrower publication types that PubMed's [Publication Type] search explodes to, so that sample/corpus
# shares are comparable with the population counts.
EVIDENCE_STRATA: list[tuple[str, set[str]]] = [
    ("meta_analysis", {"Meta-Analysis"}),
    ("systematic_review", {"Systematic Review"}),
    ("rct", {"Randomized Controlled Trial"}),
    ("clinical_trial_other", {"Clinical Trial", "Clinical Trial, Phase I", "Clinical Trial, Phase II",
                              "Clinical Trial, Phase III", "Clinical Trial, Phase IV", "Controlled Clinical Trial",
                              "Pragmatic Clinical Trial", "Equivalence Trial", "Adaptive Clinical Trial"}),
    ("observational", {"Observational Study"}),
    ("guideline", {"Practice Guideline", "Guideline"}),
    ("review", {"Review", "Scoping Review"}),
    ("case_report", {"Case Reports"}),
]


def evidence_stratum(publication_types: Iterable[str]) -> str:
    pts = set(publication_types)
    for name, members in EVIDENCE_STRATA:
        if pts & members:
            return name
    return "other"


def composition(df: pd.DataFrame) -> pd.DataFrame:
    return df.groupby("source").agg(
        docs=("doc_id", "count"), words_median=("n_words", "median"),
        words_p10=("n_words", lambda s: s.quantile(.1)), words_p90=("n_words", lambda s: s.quantile(.9)),
        words_max=("n_words", "max"), sections_median=("n_sections", "median"),
        empty_text=("n_words", lambda s: int((s == 0).sum())))


_FDA_KEY_SECTIONS = {"indications_and_usage", "dosage_and_administration", "contraindications", "adverse_reactions"}


def quality_flags(docs: list[Document]) -> dict[str, float | int]:
    pm = [d for d in docs if d.source == "pubmed"]
    fda = [d for d in docs if d.source == "openfda_label"]
    ct = [d for d in docs if d.source == "clinicaltrials"]
    ep = [d for d in docs if d.source == "europepmc_oa"]
    words = lambda d: len(_TOKEN.findall(d.text))  # noqa: E731  # same tokenization as docs_to_frame()
    out: dict[str, float | int] = {}
    if pm:
        out.update({
            "pubmed_retracted": sum(bool(d.extra.get("is_retracted")) for d in pm),
            "pubmed_with_correction_or_concern": sum(bool(d.extra.get("has_correction")) for d in pm),
            "pubmed_structured_abstract_share": round(sum(bool(d.extra.get("structured_abstract")) for d in pm) / len(pm), 3),
            "pubmed_abstract_lt_50_words": sum(words(d) < 50 for d in pm),
            "pubmed_abstract_recovered_from_OtherAbstract": sum(d.extra.get("abstract_source") == "OtherAbstract" for d in pm),
            "pubmed_empty_abstract": sum(words(d) == 0 for d in pm),
            "pubmed_non_english": sum((d.language or "") != "eng" for d in pm),
        })
    if fda:
        out.update({
            "openfda_labels_missing_any_key_section": sum(not _FDA_KEY_SECTIONS <= {s.name for s in d.sections} for d in fda),
            "openfda_otc_labels": sum((d.extra.get("product_type") or "").upper().startswith("HUMAN OTC") for d in fda),
            "openfda_unstructured_labels_fallback": sum(bool(d.extra.get("unstructured_label")) for d in fda),
            "openfda_empty_text": sum(not d.text.strip() for d in fda),
            "openfda_combination_product_labels": sum(bool(d.extra.get("is_combination", d.extra.get("n_generic_names", 1) > 1)) for d in fda),
        })
    if ct:
        out.update({
            "clinicaltrials_with_posted_results": sum(bool(d.extra.get("has_results")) for d in ct),
            "clinicaltrials_without_detailed_description": sum(len(d.sections) < 2 for d in ct),
            "clinicaltrials_with_result_publication_pmid": sum(bool(d.identifiers.get("result_pmids")) for d in ct),
        })
    if ep:
        out.update({
            "europepmc_oa_without_body": sum(not d.extra.get("has_body") for d in ep),
            "europepmc_oa_empty_text": sum(not d.text.strip() for d in ep),
            "europepmc_oa_tables_total": sum(int(d.extra.get("n_tables", 0)) for d in ep),
        })
    return out


def pubtator_coverage(entities: list[EntityAnnotation], relations: list[RelationAnnotation],
                      pubmed_docs: list[Document], drug_names: set[str]) -> dict:
    ent_df = pd.DataFrame([e.model_dump() for e in entities])
    n = len(pubmed_docs)
    if ent_df.empty:
        return {"docs": n, "with_any_entity": 0}
    by_doc = ent_df.groupby("pmid").entity_type.agg(set)
    chem_text = ent_df[ent_df.entity_type == "Chemical"].groupby("pmid").text.agg(lambda s: {t.lower() for t in s})
    chem = ent_df.entity_type == "Chemical"
    return {
        "docs": n, "with_any_entity": int(len(by_doc)),
        "with_chemical": int(by_doc.map(lambda s: "Chemical" in s).sum()),
        "with_disease": int(by_doc.map(lambda s: "Disease" in s).sum()),
        "with_chemical_and_disease": int(by_doc.map(lambda s: {"Chemical", "Disease"} <= s).sum()),
        "mentioning_in_scope_drug_exact": int(chem_text.map(lambda s: bool(s & drug_names)).sum()),
        "chemical_mentions_with_identifier_share": round(float(ent_df.loc[chem, "identifier"].notna().mean()), 3) if chem.any() else None,
        "entity_types": ent_df.entity_type.value_counts().to_dict(),
        "relation_types": dict(Counter(r.relation_type for r in relations)),
    }
