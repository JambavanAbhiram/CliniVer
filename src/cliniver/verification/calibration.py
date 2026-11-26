"""Calibration, confidence and abstention metrics (NB07, research_foundation §9.7).

* ``fit_temperature``   — one scalar T minimizing NLL of softmax(z / T) on a calibration split (Guo et al., 2017).
* ``ece``               — expected calibration error of the top-label confidence, **equal-mass** bins.
* ``brier``             — multiclass Brier score.
* ``risk_coverage`` / ``aurc`` — sort by confidence (high first); risk = error rate among the answered items;
                          AURC = mean risk over all coverage levels (lower is better).
* ``selective``         — risk, false-support rate and coverage at a confidence threshold.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize_scalar

LABELS = ("SUP", "CON", "INS")


def softmax(z: np.ndarray, t: float = 1.0) -> np.ndarray:
    z = z / t
    e = np.exp(z - z.max(1, keepdims=True))
    return e / e.sum(1, keepdims=True)


def fit_temperature(logits: np.ndarray, y: np.ndarray) -> float:
    def nll(t):
        p = softmax(logits, t)
        return -np.mean(np.log(p[np.arange(len(y)), y] + 1e-12))
    return float(minimize_scalar(nll, bounds=(0.05, 20.0), method="bounded").x)


def ece(conf: np.ndarray, correct: np.ndarray, bins: int = 15) -> float:
    order = np.argsort(conf)
    out = 0.0
    for part in np.array_split(order, bins):
        if len(part):
            out += len(part) / len(conf) * abs(conf[part].mean() - correct[part].mean())
    return float(out)


def brier(p: np.ndarray, y: np.ndarray) -> float:
    oh = np.zeros_like(p); oh[np.arange(len(y)), y] = 1
    return float(np.mean(np.sum((p - oh) ** 2, axis=1)))


def risk_coverage(conf: np.ndarray, correct: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    order = np.argsort(-conf, kind="stable")
    err = 1 - correct[order].astype(float)
    n = np.arange(1, len(conf) + 1)
    return n / len(conf), np.cumsum(err) / n


def aurc(conf: np.ndarray, correct: np.ndarray) -> float:
    return float(risk_coverage(conf, correct)[1].mean())


def selective(conf: np.ndarray, pred: np.ndarray, gold: np.ndarray, threshold: float) -> dict:
    """Metrics on the items answered (conf ≥ threshold). False support: predicted SUP with gold CON/INS."""
    keep = conf >= threshold
    n = int(keep.sum())
    fs_den = max(1, int(np.isin(gold[keep], ["CON", "INS"]).sum()))
    return {"coverage": n / len(conf), "risk": float(np.mean(pred[keep] != gold[keep])) if n else 0.0,
            "false_support_rate": float(np.sum((pred[keep] == "SUP") & np.isin(gold[keep], ["CON", "INS"])) / fs_den) if n else 0.0,
            "answered": n}


def threshold_for_coverage(conf: np.ndarray, coverage: float) -> float:
    return float(np.quantile(conf, 1 - coverage))
