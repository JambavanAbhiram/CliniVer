"""PubTator 3.0 entity and relation annotations for PubMed articles (graph construction input, ADR-0003)."""

from __future__ import annotations

import json

from cliniver.config.data_config import PubTatorCfg
from cliniver.ingestion.http import RateLimitedClient, RawStore, cached_get, parallel_map
from cliniver.schemas.document import EntityAnnotation, RelationAnnotation


class PubTatorClient:
    def __init__(self, cfg: PubTatorCfg, store: RawStore, workers: int = 1) -> None:
        self.cfg = cfg
        self.store = store
        self.workers = cfg.in_flight(workers)  # NCBI service: config may cap in-flight requests
        self.http = RateLimitedClient("pubtator", cfg.min_interval_s, max_in_flight=self.workers)

    # PubTator answers HTTP 400 with this body when NONE of the requested PMIDs is available
    # (verified 2026-11-20: a batch with at least one available PMID returns 200 with that subset).
    _NONE_AVAILABLE = "Could not retrieve publications"

    def _fetch_batch(self, batch: list[str], name: str) -> list[dict]:
        body = cached_get(self.http, self.store, name, "json", f"{self.cfg.base_url}/publications/export/biocjson",
                          {"pmids": ",".join(batch)}, not_found_ok=True, not_found_status=(400, 404),
                          not_found_text=self._NONE_AVAILABLE)
        if body is None:
            return []
        payload = json.loads(body)
        return payload.get("PubTator3", []) if isinstance(payload, dict) else payload

    def export(self, pmids: list[str], cache_name: str) -> list[dict]:
        def one(i: int) -> list[dict]:
            return self._fetch_batch(pmids[i:i + self.cfg.batch], f"biocjson_{cache_name}_{i // self.cfg.batch:04d}")

        batches = parallel_map(one, range(0, len(pmids), self.cfg.batch), self.workers)
        return [doc for b in batches for doc in b]

    def export_parsed(self, pmids: list[str], cache_name: str
                      ) -> tuple[list[EntityAnnotation], list[RelationAnnotation], set[str]]:
        """Like ``export`` but parses each batch as it arrives (bounded memory at corpus scale).

        Returns (entities, relations, PMIDs PubTator returned a document for).
        """
        def one(i: int):
            docs = self._fetch_batch(pmids[i:i + self.cfg.batch], f"biocjson_{cache_name}_{i // self.cfg.batch:05d}")
            ents, rels, seen = [], [], set()
            for d in docs:
                e, r = parse_bioc(d)
                ents += e
                rels += r
                seen.add(str(d.get("pmid") or d["id"]).split("|")[0])
            return ents, rels, seen

        parts = parallel_map(one, range(0, len(pmids), self.cfg.batch), self.workers)
        return ([e for p in parts for e in p[0]], [r for p in parts for r in p[1]],
                set().union(*(p[2] for p in parts)) if parts else set())


def parse_bioc(doc: dict) -> tuple[list[EntityAnnotation], list[RelationAnnotation]]:
    pmid = str(doc.get("pmid") or doc["id"]).split("|")[0]
    ents: list[EntityAnnotation] = []
    for passage in doc.get("passages", []):
        ptype = passage.get("infons", {}).get("type", "unknown")
        for ann in passage.get("annotations", []):
            inf = ann.get("infons", {})
            for loc in ann.get("locations", []):
                ents.append(EntityAnnotation(
                    pmid=pmid, passage_type=ptype, entity_type=inf.get("type", "unknown"),
                    identifier=str(inf["identifier"]) if inf.get("identifier") not in (None, "-") else None,
                    text=ann.get("text", ""), offset=int(loc["offset"]), length=int(loc["length"])))
    rels: list[RelationAnnotation] = []
    for rel in doc.get("relations", []):
        inf = rel.get("infons", {})
        r1, r2 = inf.get("role1", {}), inf.get("role2", {})
        score = inf.get("score")
        rels.append(RelationAnnotation(
            pmid=pmid, relation_type=inf.get("type", "unknown"),
            entity1=str(r1.get("identifier") if isinstance(r1, dict) else r1),
            entity2=str(r2.get("identifier") if isinstance(r2, dict) else r2),
            score=float(score) if score not in (None, "") else None))
    return ents, rels
