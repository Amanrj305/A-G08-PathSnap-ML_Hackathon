# First evaluation: show these items first

Your teacher scheduled the first 10-mark evaluation for **18 November 2026, 11:30 AM IST**. The team's requested preparation target is **11:00 AM**.

1. Open `output/phase1/PathSnap_First_Evaluation.pdf`.
2. Open `results/dataset_inventory.csv`: 80 training trips and 10 student-test trips. Explain the file types, timestamp audit, missing-value checks and unusual reference distances.
3. Open `pathsnap/features.py`: explain projection to metres, eight directed candidates and the 18 features.
4. Open `results/baseline/model_comparison.csv`: compare six trained models and explain why observation Top-1 matters more than negative-heavy candidate accuracy.
5. Open `results/baseline/cross_validation.json`: demonstrate that entire trips stay in one fold.
6. Open `artifacts/model_metadata.json`: this identifies the model chosen by grouped validation, its measured metrics and software versions.
7. Run a real prediction (no route labels are read):

```powershell
cd D:\ML_HAKATHON\PathSnap_1_Preprocessing\PathSnap_1_Preprocessing
python -m pytest tests/test_core.py -v
```

8. Open `output/phase1/development_map.html` in a browser. Layer controls show GPS, network, reference and predictions. This development map uses student-test references; evaluator UI must never expose evaluator references.

## Be ready to explain

- `.route` is an ordered list of arc IDs, NOT one label per GPS point. The project derives point labels by monotone alignment, and reports that limitation.
- Every candidate row is a binary match/non-match example; Top-1 chooses one candidate per observation.
- A candidate miss is counted as a Top-1 failure. The correct arc is never inserted into candidates using a label.
- Feature extraction never reads `.route`. Training and scoring read labels in separate modules.
- Three-fold GroupKFold prevents nearby samples from the same trip entering both fitting and validation. Scaling happens within each fold.
- Feature windows use future GPS too: this implementation processes full trips offline.
- The first-phase fit caps observations per training trip for runtime; the exact sample count and seed are recorded. All student-test points are scored.
- No artificial training curves are provided for these tree/probabilistic models.
- Specificity means the proportion of negative candidates correctly rejected. It is not route accuracy.

## Four teammates

| Teammate | First-phase responsibility | Later responsibility |
|---|---|---|
| 1 | Raw format, audit, projection, candidate generation and features | Data contracts and reproducibility |
| 2 | Six model runs, grouped CV, selection and metrics | Bayesian tuning and local improvement |
| 3 | Explain classifier prediction and artifact loading | HMM/Viterbi + FastAPI on Laptop 1 |
| 4 | Maps, report, comparison figures, demo walkthrough | UI on Laptop 2 and final submission evidence |

These are ownership assignments, not claims that teammates have already authored or reviewed the work. Review and understand your part before presenting. Disclose AI assistance according to event rules.

## Timeline after this checkpoint

- 11:30 AM: first evaluation (10 marks).
- Next: tune using training-trip CV, compare HMM decoding, integrate API/UI and test on two laptops.
- Before 3:00 PM: capture actual two-laptop evidence, finalize the consolidated PDF, publish the authorized repository and submit through the teacher's form.
- 4:30 PM: form closes; 5:00 PM: event ends. Aim for 3:00 PM, not the closing time.

SSH host details and teammate names are intentionally deferred until version 0 is ready, as requested.
