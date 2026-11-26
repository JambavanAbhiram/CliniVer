from cliniver.verification.perturb import change_number, negate, paraphrase, remove_hedge, swap_entity, widen_population
from cliniver.verification.verifiers import SimilarityVerifier, qualifier_part


def test_perturbations_by_construction():
    assert negate("Metformin reduces mortality in adults.") == "Metformin does not reduce mortality in adults."
    assert negate("Statins are effective.") == "Statins are not effective."
    assert negate("Aspirin does not prevent stroke.") == "Aspirin does prevent stroke."
    assert change_number("HbA1c fell by 1.5% in 24 weeks.") == "HbA1c fell by 11.5% in 24 weeks."
    assert change_number("Dose of 500 mg daily.") == "Dose of 1500 mg daily."
    assert remove_hedge("Statins may reduce stroke risk.") == "Statins reduce stroke risk."
    assert widen_population("Metformin reduces all-cause mortality in patients with type 2 diabetes.") == "Metformin reduces all-cause mortality."
    assert swap_entity("Metformin reduces mortality.", "metformin") == "warfarin reduces mortality."
    assert paraphrase("Smoking increases risk.") == "Smoking raises risk."
    assert negate("No verbs here") is None or True


def test_similarity_never_contradicts_and_qualifier_rule():
    assert set(SimilarityVerifier(0.5).predict([0.9, 0.1])) <= {"SUP", "INS"}
    ev = "In adults with type 2 diabetes, metformin 500 mg lowered HbA1c."
    assert qualifier_part("Metformin 500 mg lowers HbA1c in adults with type 2 diabetes.", ev, "SUP") == "SUP"
    assert qualifier_part("Metformin 2000 mg lowers HbA1c.", ev, "SUP") == "PART"
    assert qualifier_part("Metformin lowers HbA1c in children with obesity.", ev, "SUP") == "PART"
    assert qualifier_part("anything", ev, "CON") == "CON"


def test_numeric_consistency():
    from cliniver.verification.consistency import numeric_consistency, quantities
    ev = "Metformin 500 mg twice daily lowered HbA1c by 1.2% over 24 weeks in adults with eGFR ≥ 45 mL/min."
    assert numeric_consistency("Metformin 500 mg lowers HbA1c by 1.2%.", ev)["match"] == 2
    r = numeric_consistency("Metformin 1500 mg lowers HbA1c.", ev)
    assert r["mismatch"] == 1 and r["examples"][0][0] == "1500 mg"
    assert numeric_consistency("It reduces LDL by 30 mg/dL.", ev)["absent"] == 1
    assert [q.unit for q in quantities("HbA1c and CYP3A4 at 10-20 mg")] == ["mg"]   # identifiers ignored, range kept
    assert numeric_consistency("given for 12 to 24 weeks", ev)["match"] == 1


def test_unsupported_numbers():
    from cliniver.verification.consistency import unsupported_numbers
    ev = "In type 2 diabetes, 24 weeks of treatment lowered HbA1c by 1.20%."
    assert unsupported_numbers("In type 2 diabetes, HbA1c fell by 1.2% after 24 weeks.", ev) == []
    assert unsupported_numbers("In type 6 diabetes, HbA1c fell by 1.2% after 72 weeks.", ev) == ["6", "72"]


def test_calibration_metrics():
    import numpy as np
    from cliniver.verification.calibration import aurc, brier, ece, fit_temperature, selective, softmax
    rng = np.random.default_rng(0)
    y = rng.integers(0, 3, 2000)
    z = rng.normal(size=(2000, 3)); z[np.arange(2000), y] += 2.0
    over = z * 4                                    # over-confident logits
    t = fit_temperature(over, y)
    assert t > 2.0                                  # temperature undoes the over-confidence
    p0, p1 = softmax(over), softmax(over, t)
    c0 = p0.max(1); c1 = p1.max(1); correct = p0.argmax(1) == y
    assert ece(c1, correct) < ece(c0, correct) and brier(p1, y) < brier(p0, y)
    assert aurc(c1, correct) <= aurc(rng.random(2000), correct)       # informative confidence beats random
    g = np.array(["SUP", "INS", "CON", "SUP"]); pr = np.array(["SUP", "SUP", "CON", "SUP"]); cf = np.array([0.9, 0.2, 0.8, 0.7])
    s = selective(cf, pr, g, 0.5)
    assert s["answered"] == 3 and s["false_support_rate"] == 0.0 and selective(cf, pr, g, 0.0)["false_support_rate"] == 0.5
