"""Numeric consistency between a claim and its evidence (NB07; addresses the NB06 finding that no verifier
detected changed numbers/doses in more than 12% of cases).

For every quantity in the claim (number + unit, or a percentage) the check looks for the same unit in the evidence:
* ``match``     — the evidence contains the same value with the same unit (tolerance ``rel_tol``)
* ``mismatch``  — the evidence contains the same unit but only different values (e.g. claim 1500 mg, evidence 500 mg)
* ``absent``    — the unit does not occur in the evidence (the number is unsupported, not contradicted)

Numbers inside identifiers ("HbA1c", "T2D", "CYP3A4") are ignored; ranges count as both endpoints. The rule is a
signal for calibration and abstention, never a verdict by itself (a mismatch can be a legitimate different subgroup).
"""

from __future__ import annotations

import re
from dataclasses import dataclass

_UNITS = {
    "mg": "mg", "milligram": "mg", "milligrams": "mg", "g": "g", "gram": "g", "grams": "g", "mcg": "mcg", "µg": "mcg", "μg": "mcg",
    "kg": "kg", "ml": "ml", "l": "l", "mmol/l": "mmol/l", "mg/dl": "mg/dl", "mmhg": "mmhg", "mm hg": "mmhg", "iu": "iu",
    "units": "units", "unit": "units", "%": "%", "percent": "%", "years": "years", "year": "years", "months": "months",
    "month": "months", "weeks": "weeks", "week": "weeks", "days": "days", "day": "days", "hours": "hours", "hour": "hours",
    "h": "hours", "ml/min": "ml/min", "ml/min/1.73 m2": "ml/min", "ml/min/1.73m2": "ml/min", "mg/kg": "mg/kg", "bpm": "bpm",
}
_NUM = r"(?<![A-Za-z\d.])(\d+(?:[.,]\d+)?)(?:\s?(?:-|–|to)\s?(\d+(?:[.,]\d+)?))?"
_UNIT_ALT = "|".join(sorted((re.escape(u) for u in _UNITS), key=len, reverse=True))
_QTY = re.compile(rf"{_NUM}\s?({_UNIT_ALT})(?![A-Za-z])", re.I)


@dataclass(frozen=True)
class Quantity:
    values: tuple[float, ...]
    unit: str
    text: str


def quantities(text: str) -> list[Quantity]:
    out = []
    for m in _QTY.finditer(text):
        vals = tuple(float(x.replace(",", ".")) for x in (m.group(1), m.group(2)) if x)
        out.append(Quantity(vals, _UNITS[m.group(3).lower()], m.group(0)))
    return out


def numeric_consistency(claim: str, evidence: str, rel_tol: float = 0.02) -> dict:
    """Returns counts of claim quantities that match / mismatch / are absent in the evidence, plus examples."""
    ev = quantities(evidence)
    res = {"match": 0, "mismatch": 0, "absent": 0, "examples": []}
    for q in quantities(claim):
        same_unit = [e for e in ev if e.unit == q.unit]
        if not same_unit:
            res["absent"] += 1; continue
        ok = any(abs(v - w) <= rel_tol * max(abs(v), abs(w), 1e-9) for e in same_unit for v in q.values for w in e.values)
        if ok:
            res["match"] += 1
        else:
            res["mismatch"] += 1
            res["examples"].append((q.text, [e.text for e in same_unit][:3]))
    return res


_BARE = re.compile(r"(?<![A-Za-z\d.,])(\d+(?:\.\d+)?)(?![A-Za-z\d])")


def bare_numbers(text: str) -> set[str]:
    """Standalone numbers (not inside identifiers such as HbA1c / CYP3A4 / COVID-19's 19 is kept: it is standalone)."""
    out = set()
    for m in _BARE.finditer(text):
        v = m.group(1)
        out.add(v.rstrip("0").rstrip(".") if "." in v else v)
    return out


def unsupported_numbers(claim: str, evidence: str) -> list[str]:
    """Claim numbers that occur nowhere in the evidence (as standalone numbers). A weak signal: evidence may state
    the same fact with different numbers (e.g. 50% vs one half)."""
    ev = bare_numbers(evidence)
    return sorted(n for n in bare_numbers(claim) if n not in ev)
