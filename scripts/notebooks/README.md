# Notebook builders

The research notebooks in `notebooks/` are built from these scripts and then executed headlessly. The executed notebook is the research artifact; the builder makes its construction reproducible and reviewable as plain Python.

| Notebook | Builder |
|---|---|
| `01_data_acquisition_and_analysis.ipynb` | `nb01/build_nb01.py` (with `nb01_discussion.py`, `stage_b_section.py`, `stage_b_discussion.py`) |
| `02_document_processing_and_chunking.ipynb` | `build_nb02.py` (+ `nb02_discussion.py`, written after execution from the outputs) |
| `03_llm_responses_and_claim_extraction.ipynb` | `build_nb03.py` (+ `nb03_discussion.py`). LLM calls replay from `data/interim/<snapshot>/llm_calls/`, so re-execution needs no generation |
| `04_claim_contextualization.ipynb` | `build_nb04.py` (+ `nb04_discussion.py`); LLM calls replay from the call log |
| `05_retrieval.ipynb` | `build_nb05.py` (+ `nb05_discussion.py`); embeddings cached in `data/interim/<snapshot>/embeddings/` |
| `06_claim_evidence_verification.ipynb` | `build_nb06.py` (+ `nb06_discussion.py`); the fine-tuned verifier is cached in `data/interim/<snapshot>/models/` |
| `07_confidence_calibration_abstention.ipynb` | `build_nb07.py` (+ `nb07_discussion.py`); datasets via `cliniver.verification.datasets` |
| `09_answer_generation.ipynb` | `build_nb09.py` (+ `nb09_discussion.py`); owner-approved 9th notebook; rounds 1–3 (EXP-09-001…003) |
| `08_end_to_end_evaluation.ipynb` | `build_nb08.py` (+ `nb08_discussion.py`); Part B `nb08_partb.py` (+ `nb08b_discussion.py`), Part C `nb08_partc.py` (+ `nb08c_discussion.py`). Part C always measures live (fresh call logs) |

```powershell
# build, then execute in place (outputs saved even if a cell fails; exit status 1 on any error)
.venv\Scripts\python.exe scripts\notebooks\build_nb02.py notebooks\02_document_processing_and_chunking.ipynb
.venv\Scripts\python.exe scripts\notebooks\run_notebook.py notebooks\02_document_processing_and_chunking.ipynb
```

NB02–NB08 support a quick code check: set `CLINIVER_SMOKE=1` to run on small subsets under a `-smoke` experiment id (never used for results).

Discussion cells are written **after** execution, from the actual outputs. Rebuilding a notebook replaces its outputs, so it must then be re-executed.
