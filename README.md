# PathSnap - version 0

GPS map matching with learned candidate probabilities and a directed-network HMM/Viterbi decoder. Source, experiments and report are built from the supplied 80 training and 10 student-test trips. Final evaluator trips are not supplied.

**Start with `docs/FIRST_EVALUATION.md` for the first 10-mark checkpoint.**

## Setup (Python 3.12 recommended)

Windows PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Linux/macOS:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

Put the supplied dataset at `data/Student folder/train/` and `data/Student folder/student_test/`. Each trip folder contains identically named `.track`, `.nodes`, `.arcs`, and (for development only) `.route` files. Raw data is ignored by Git; keep it local unless its redistribution is explicitly permitted.

## Reproduce the first evaluation

Use your virtual environment's Python in place of `python` below:

```bash
python -m scripts.prepare
python -m scripts.train
python -m scripts.phase1_report
python -m scripts.predict --classifier-only
python -m pytest -q -p no:hypothesispytest
```

Training: six syllabus models (Logistic Regression, Gaussian Naive Bayes, ID3, KNN, Random Forest and Gradient Boosting), three-fold trip-group validation, seed 35218, up to 600 observations per training trip, eight candidates, 18 features. All 19,160 student-test observations are evaluated. Set `--max-obs-per-trip 0 --folds 5 --output results/full_baseline` for a larger comparison. Use a separate project copy for new experiments to preserve the demonstrated artifact.

## Structure

- `pathsnap/data.py`, `features.py`: label-free parsing and feature engineering.
- `pathsnap/reference.py`: separate derivation of proxy labels for training/scoring.
- `pathsnap/modeling.py`, `scripts/train.py`: six reproducible model families/configurations.
- `pathsnap/decoder.py`: directed graph transitions, gap handling and Viterbi.
- `scripts/predict.py`: label-free command-line inference.
- `results/`: audit, configurations, cross-validation records, metrics and saved baselines.
- `artifacts/`: selected inference model and metadata.
- `output/phase1/`: first-evaluation report, figures and development Folium map.
- `docs/`: demo and teammate instructions.
- `tests/`: meaningful checks of spatial retrieval, label isolation, alignment and metrics.

## What the scores mean

The supplied ordered route has no point timestamps. We derive per-GPS proxy labels using monotone geometric alignment. These labels are useful for development, but are not independent per-point ground truth. The final evaluator's scoring definition may differ.

Candidate binary accuracy, precision, recall, specificity and macro F1 use match/non-match labels and threshold 0.5. Observation Top-1 selects one arc per GPS point; it is the main route-selection metric. Candidate misses remain failures. Final model selection uses mean group-CV Top-1, not training accuracy or the unseen evaluator partition.

Features do not use route labels, global arc IDs or trip IDs. The decoder accepts only raw inference inputs and classifier probabilities. Reference data appears only in offline development maps and the separate server-side scorer.

## Academic provenance

AI assistance was used to implement, execute, validate and document this project. Actual saved experiment outputs support reported results. Team members should review, reproduce and explain their assigned work and disclose assistance according to the event policy. Do not claim experiments, network deployments or independent authorship that did not occur.

## Technical references

- [scikit-learn GroupKFold](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GroupKFold.html)
- [Optuna TPE sampler](https://optuna.readthedocs.io/en/stable/reference/samplers/generated/optuna.samplers.TPESampler.html)
- [Newson and Krumm, HMM map matching (2009)](https://www.microsoft.com/en-us/research/publication/hidden-markov-map-matching-noise-sparseness/)
- The five supplied PathSnap guideline PDFs.
