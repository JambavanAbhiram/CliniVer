"""Drug normalization sources: RxNorm / RxClass (NLM RxNav) and PubChem.

These produce ``Concept`` records (identity, synonyms, classes) for entity linking and the
concept graph. They are never treated as evidence. Note: RxNav's drug–drug interaction API was
discontinued on 2024-01-02 and is therefore not used.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone

from cliniver.config.data_config import PubChemCfg, RxNavCfg
from cliniver.ingestion.http import RateLimitedClient, RawStore, cached_get
from cliniver.schemas.document import Concept


class RxNavClient:
    def __init__(self, cfg: RxNavCfg, store: RawStore, workers: int = 1) -> None:
        self.cfg = cfg
        self.store = store
        self.workers = workers
        self.http = RateLimitedClient("rxnav", cfg.min_interval_s, max_in_flight=workers)

    def _get(self, name: str, path: str, params: dict | None = None) -> dict:
        return json.loads(cached_get(self.http, self.store, name, "json", f"{self.cfg.base_url}/{path}", params))

    def concepts_for(self, drug: str, *, snapshot: str) -> list[Concept]:
        """Ingredient concept (with brand names) plus its drug classes. Empty if the name is unknown."""
        ids = self._get(f"rxcui_{drug}", "rxcui.json", {"name": drug, "search": "2"}).get("idGroup", {})
        rxcuis = ids.get("rxnormId") or []
        if not rxcuis:
            return []
        rxcui = rxcuis[0]
        now = datetime.now(timezone.utc)
        related = self._get(f"brands_{drug}", f"rxcui/{rxcui}/related.json", {"tty": "BN"})
        brands = [c for g in (related.get("relatedGroup", {}).get("conceptGroup") or [])
                  for c in (g.get("conceptProperties") or [])]
        out: list[Concept] = []
        rows = []
        for src in self.cfg.class_sources:
            data = self._get(f"classes_{drug}_{src}", "rxclass/class/byRxcui.json",
                             {"rxcui": rxcui, "relaSource": src})
            rows += [(src, item) for item in (data.get("rxclassDrugInfoList") or {}).get("rxclassDrugInfo", [])]
        # RxClass also returns classes of *combination* products containing the drug. Keep only rows
        # where the drug itself is the member (avoids false graph edges such as metformin → "DPP-4
        # inhibitor" via sitagliptin/metformin). Fallback: an ingredient marketed ONLY in combination
        # (e.g. sacubitril → sacubitril/valsartan) keeps its combination classes, flagged as such.
        own = [(s, i) for s, i in rows if i["minConcept"]["rxcui"] == rxcui]
        via_combination = not own
        selected = own or rows
        classes: list[Concept] = []
        for src, item in selected:
            cls = item["rxclassMinConceptItem"]
            classes.append(Concept(
                concept_id=f"rxclass:{cls['classType']}:{cls['classId']}", source="rxclass", kind="drug_class",
                name=cls["className"], query_name=drug, retrieved_at=now, snapshot=snapshot,
                extra={"class_type": cls["classType"], "rela": item.get("rela"), "rela_source": src,
                       "member_rxcui": rxcui, "via_combination": via_combination,
                       "combination_member": item["minConcept"]["name"] if via_combination else None}))
        out.append(Concept(
            concept_id=f"rxnorm:{rxcui}", source="rxnorm", kind="drug", name=drug, query_name=drug,
            synonyms=sorted({b["name"] for b in brands}),
            xrefs={"brand_rxcui": [b["rxcui"] for b in brands],
                   "class_ids": sorted({c.concept_id for c in classes})},
            retrieved_at=now, snapshot=snapshot, extra={"all_rxcuis_for_name": rxcuis}))
        return out + classes


class PubChemClient:
    def __init__(self, cfg: PubChemCfg, store: RawStore, workers: int = 1) -> None:
        self.cfg = cfg
        self.store = store
        self.workers = workers
        self.http = RateLimitedClient("pubchem", cfg.min_interval_s, max_in_flight=workers)

    def concept_for(self, drug: str, *, snapshot: str) -> Concept | None:
        """CID + synonyms. Returns None for names PubChem's compound database does not hold
        (typically biologics such as monoclonal antibodies or insulin analogues)."""
        body = cached_get(self.http, self.store, f"cids_{drug}", "json",
                          f"{self.cfg.base_url}/compound/name/{drug}/cids/JSON", not_found_ok=True)
        if body is None:
            return None
        cid_list = json.loads(body).get("IdentifierList", {}).get("CID", [])
        if not cid_list:
            return None
        cid = cid_list[0]
        syn = json.loads(cached_get(self.http, self.store, f"synonyms_{cid}", "json",
                                    f"{self.cfg.base_url}/compound/cid/{cid}/synonyms/JSON"))
        names = syn["InformationList"]["Information"][0].get("Synonym", [])
        return Concept(concept_id=f"pubchem:{cid}", source="pubchem", kind="drug", name=drug, query_name=drug,
                       synonyms=names[: self.cfg.max_synonyms], xrefs={"cid": [str(c) for c in cid_list]},
                       retrieved_at=datetime.now(timezone.utc), snapshot=snapshot,
                       extra={"n_synonyms_total": len(names), "n_cids_for_name": len(cid_list)})
