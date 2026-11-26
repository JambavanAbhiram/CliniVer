# CliniVer

**Clinical AI Hallucination Detection & Evidence Verification Engine** — research prototype.

CliniVer takes a clinical question and an AI-generated answer. It breaks the answer into individual clinical claims, retrieves evidence for each claim from a dated, openly licensed biomedical corpus, and reports whether each claim is **Supported**, **Partially Supported**, **Contradicted**, or has **Insufficient Evidence**. It shows the evidence, the sources, the dates, and a confidence assessment, and abstains when it cannot decide reliably.

> ⚠️ **Not a medical device. Not medical advice.** CliniVer is an academic research tool. Verdicts are relative to a corpus snapshot. "Insufficient evidence" never means "false". Always consult a qualified clinician.

## Status

Research prototype. All nine notebooks (01–09) have been executed; experiment records (configuration, versions, hardware, results) are in `artifacts/experiments/`. A local Streamlit interface generates evidence-grounded answers and verifies them claim by claim (`scripts/run_app.py`). No clinical validation has been done.

## Research design

- **Domain:** adult cardiometabolic pharmacotherapy.
- **Evidence sources:** PubMed, PMC Open Access, openFDA/DailyMed, ClinicalTrials.gov. Normalization and graph construction use PubTator 3, RxNorm, MeSH and PubChem. WHO GHO provides population statistics.
- **Retrieval study:** sparse (BM25) + dense + concept-graph retrieval, kept only if experiments justify it.
- **Models:** open-weight only, run locally on an 8 GB GPU. Zero cost.

## Setup (Windows, Python 3.12)

```powershell
py -3.12 scripts/setup_environment.py          # creates .venv, installs PyTorch (CUDA 13.0) + requirements
copy .env.example .env                          # then add NCBI_API_KEY / NCBI_EMAIL
.venv\Scripts\python.exe scripts\validate_environment.py
.venv\Scripts\python.exe -m pytest
```

On a machine without an NVIDIA GPU: `py -3.12 scripts/setup_environment.py --device cpu`.
In VS Code, select the Jupyter kernel **"CliniVer (Python 3.12)"**.

## Repository layout

```
configs/        YAML configuration (validated by cliniver.config)
notebooks/      the research notebooks (01–09)
scripts/        environment setup and validation, app launcher, notebook builders
src/cliniver/   the Python package (shared, tested code)
tests/          unit / integration / evaluation tests
data/, artifacts/  data and generated artifacts (git-ignored except manifests and experiment results)
```

## License

To be decided. Third-party data is **not** redistributed. The repository contains identifiers and acquisition scripts only.
