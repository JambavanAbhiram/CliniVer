"""The NB07 confidence model as a reusable component (used by NB08 and the future service).

Fitted on a calibration split (HealthVer dev): temperature T, a logistic multi-signal model predicting "the verifier
is correct", the second-opinion routing threshold (25th percentile) and the 80%-coverage abstention threshold.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from cliniver.evaluation.extraction import content_words
from cliniver.verification.calibration import fit_temperature, softmax, threshold_for_coverage
from cliniver.verification.consistency import numeric_consistency, unsupported_numbers

LAB = ["SUP", "CON", "INS"]
SIGNALS = ["verifier_conf", "margin", "entropy", "unit_mismatch", "unsupported_numbers", "claim_words", "evidence_words", "lexical_overlap"]


def features(claims: list[str], evidence: list[str], z: np.ndarray, t: float) -> pd.DataFrame:
    p = softmax(z, t); srt = np.sort(p, 1)
    rows = []
    for c, e, pi, si in zip(claims, evidence, p, srt):
        nc = numeric_consistency(c, e); cw = content_words(c)
        rows.append({"verifier_conf": si[-1], "margin": si[-1] - si[-2], "entropy": float(-(pi * np.log(pi + 1e-12)).sum()),
                     "unit_mismatch": int(nc["mismatch"] > 0), "unsupported_numbers": len(unsupported_numbers(c, e)),
                     "claim_words": len(c.split()), "evidence_words": len(e.split()),
                     "lexical_overlap": len(cw & content_words(e)) / max(1, len(cw))})
    return pd.DataFrame(rows)


@dataclass
class ConfidenceModel:
    t: float
    lr: LogisticRegression
    mu: pd.Series
    sd: pd.Series
    route_threshold: float
    abstain_threshold: float

    @classmethod
    def fit(cls, nli, items: list[dict], route_quantile: float = 0.25, coverage: float = 0.80) -> "ConfidenceModel":
        claims, ev = [x["claim"] for x in items], [x["evidence"] for x in items]
        y = np.array([LAB.index(x["label"]) for x in items])
        z = nli.logits(ev, claims)
        t = fit_temperature(z, y)
        f = features(claims, ev, z, t)
        mu, sd = f[SIGNALS].mean(), f[SIGNALS].std().replace(0, 1)
        lr = LogisticRegression(max_iter=1000).fit(((f[SIGNALS] - mu) / sd).values, (softmax(z).argmax(1) == y).astype(int))
        overall = lr.predict_proba(((f[SIGNALS] - mu) / sd).values)[:, 1]
        return cls(t, lr, mu, sd, float(np.quantile(overall, route_quantile)), threshold_for_coverage(overall, coverage))

    def score(self, claims: list[str], evidence: list[str], z: np.ndarray) -> tuple[list[str], np.ndarray, np.ndarray]:
        """(labels, calibrated probabilities, overall confidence) for claim–evidence pairs given NLI logits."""
        p = softmax(z, self.t)
        f = features(claims, evidence, z, self.t)
        overall = self.lr.predict_proba(((f[SIGNALS] - self.mu) / self.sd).values)[:, 1]
        return [LAB[k] for k in p.argmax(1)], p, overall
