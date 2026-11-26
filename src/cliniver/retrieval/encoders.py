"""Dense bi-encoders for NB05 (Part A/B): one interface over sentence-transformers models and MedCPT.

MedCPT (Jin et al., 2023; public domain) uses two encoders: a query encoder (max 64 tokens) and an article
encoder that takes (title, abstract) pairs (max 512 tokens); embeddings are the [CLS] vectors and are compared by
**inner product** (not normalized). The sentence-transformers models are L2-normalized (cosine).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
import torch


@dataclass
class Encoder:
    key: str
    repo: str
    revision: str
    query_prefix: str = ""
    doc_prefix: str = ""
    doc_repo: str | None = None
    doc_revision: str | None = None
    device: str = "cuda"
    normalize: bool = True
    batch_size: int = 64
    slice_size: int = 8192
    progress_log: str | None = None
    stats: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.doc_repo:  # MedCPT: separate HF encoders, CLS pooling
            from transformers import AutoModel, AutoTokenizer
            self.q_tok = AutoTokenizer.from_pretrained(self.repo, revision=self.revision)
            self.q_model = AutoModel.from_pretrained(self.repo, revision=self.revision).to(self.device).eval().half()
            self.d_tok = AutoTokenizer.from_pretrained(self.doc_repo, revision=self.doc_revision)
            self.d_model = AutoModel.from_pretrained(self.doc_repo, revision=self.doc_revision).to(self.device).eval().half()
            self.normalize = False
            self.dim = self.q_model.config.hidden_size
        else:
            from sentence_transformers import SentenceTransformer
            self.st = SentenceTransformer(self.repo, revision=self.revision, device=self.device)
            if self.device == "cuda":
                self.st.half()
            self.dim = self.st.get_sentence_embedding_dimension()
        self.n_params = sum(p.numel() for m in self._modules() for p in m.parameters())

    def _modules(self):
        return [self.q_model, self.d_model] if self.doc_repo else [self.st]

    @torch.inference_mode()
    def _cls(self, tok, model, texts, max_len, pairs=None) -> torch.Tensor:
        out = []
        for i in range(0, len(texts), self.batch_size):
            if pairs is not None:
                enc = tok(texts[i:i + self.batch_size], pairs[i:i + self.batch_size], truncation=True, padding=True,
                          max_length=max_len, return_tensors="pt").to(self.device)
            else:
                enc = tok(texts[i:i + self.batch_size], truncation=True, padding=True, max_length=max_len,
                          return_tensors="pt").to(self.device)
            # .clone(): a [CLS] *view* keeps the whole last_hidden_state alive (batch × 512 × 768) until the slice is
            # concatenated — that, not the batch size, caused the 6.9–7.3 GB peaks seen in NB05/NB06 (2026-11-21)
            out.append(model(**enc).last_hidden_state[:, 0, :].clone())
        return torch.cat(out)

    def encode_queries(self, texts: list[str]) -> torch.Tensor:
        if self.doc_repo:
            return self._cls(self.q_tok, self.q_model, texts, 64)
        return self.st.encode([self.query_prefix + t for t in texts], batch_size=128, normalize_embeddings=True,
                              convert_to_tensor=True, show_progress_bar=False)

    def encode_docs(self, texts: list[str], titles: list[str] | None = None) -> torch.Tensor:
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        t0 = time.perf_counter()
        parts = []
        for s in range(0, len(texts), self.slice_size):  # slices: bounded memory + progress visibility on large corpora
            tx = texts[s:s + self.slice_size]
            ti = titles[s:s + self.slice_size] if titles is not None else None
            if self.doc_repo:
                parts.append(self._cls(self.d_tok, self.d_model, ti if ti is not None else tx, 512, pairs=tx if ti is not None else None).half())
            else:
                body = [f"{a}. {b}" for a, b in zip(ti, tx)] if ti is not None else tx
                parts.append(self.st.encode([self.doc_prefix + b for b in body], batch_size=self.batch_size, normalize_embeddings=True,
                                            convert_to_tensor=True, show_progress_bar=False).half())
            if self.progress_log is not None:
                done = s + len(tx); el = time.perf_counter() - t0
                with open(self.progress_log, "a", encoding="utf-8") as fh:
                    fh.write(f"{time.strftime('%H:%M:%S')} {self.key} {done}/{len(texts)} docs {done / el:.0f} docs/s\n")
        emb = torch.cat(parts)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        self.stats = {"docs": len(texts), "encode_s": round(dt, 1), "docs_per_s": round(len(texts) / dt, 1),
                      "peak_vram_mb": round(torch.cuda.max_memory_allocated() / 2**20) if torch.cuda.is_available() else None}
        return emb.half()

    @staticmethod
    def search(q: torch.Tensor, d: torch.Tensor, k: int) -> tuple[np.ndarray, np.ndarray]:
        ids, scores = [], []
        for i in range(0, len(q), 256):
            s = (q[i:i + 256].to(d.dtype) @ d.T).float()
            top = torch.topk(s, min(k, d.shape[0]), dim=1)
            ids.append(top.indices.cpu().numpy()); scores.append(top.values.cpu().numpy())
        return np.vstack(ids), np.vstack(scores)

    def unload(self) -> None:
        for m in self._modules():
            m.to("cpu")
        torch.cuda.empty_cache()
