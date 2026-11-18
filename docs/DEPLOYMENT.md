# Two-laptop deployment and four teammate handoffs

SSH connection details are deferred until the local version 0 is complete, as requested. No remote deployment has been claimed or performed.

## Teammate 1 - data and features

Own `pathsnap/data.py`, `features.py`, `reference.py`, `scripts/prepare.py`, the data inventory and feature manifest. Explain the 18 features, exact nearest-segment retrieval and derived-label limitation. Handoff: `cache/` or rerunnable preparation code plus `results/data_manifest.json`. All machines must use the same feature order and candidate count from `artifacts/model_metadata.json`.

## Teammate 2 - training and tuning

Own `pathsnap/id3.py`, `modeling.py`, `metrics.py`, `scripts/train.py`, `tune.py`, `results/baseline/` and `results/tuning/`. Explain the syllabus mapping, six model results, trip-group validation and Bayesian trials. Handoff the entire `artifacts/` folder to Teammate 3; never hand off just a model without metadata.

## Teammate 3 - Laptop 1: model and API

Own `pathsnap/decoder.py`, `api/`, server-side scoring and the inference artifacts. Install Python 3.12 and create a virtual environment. From the project directory:

```powershell
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe -m uvicorn api.app:app --host 0.0.0.0 --port 8000
```

Linux uses `.venv/bin/python` in place of `.venv\Scripts\python.exe`. `deploy/start_api.ps1` or `bash deploy/start_api.sh` also saves console logs. Keep the server terminal open so evaluators can see real incoming POST requests.

The default partition is `data/Student folder/student_test` and is visibly marked development mode. When the teacher supplies the reserved evaluator data, place it in a separate folder with one eight-digit folder per trip, then set these environment variables before starting the server:

```powershell
$env:PATHSNAP_DATA_ROOT = 'C:\PathSnap\evaluator_test'
$env:PATHSNAP_MODE = 'evaluation'
.venv\Scripts\python.exe -m uvicorn api.app:app --host 0.0.0.0 --port 8000
```

Linux equivalent:

```bash
export PATHSNAP_DATA_ROOT=/home/USER/PathSnap/evaluator_test
export PATHSNAP_MODE=evaluation
.venv/bin/python -m uvicorn api.app:app --host 0.0.0.0 --port 8000
```

Prediction reads only `.track`, `.nodes` and `.arcs`. Optional `.route` is read by the isolated scorer after inference. It never enters features/model/decoder and is never returned to the UI. If the teacher has an official timestamp-to-arc scorer, adapt `score_after_prediction` to that scoring contract. Current scoring derives point labels from the supplied ordered route and says so explicitly. Without a `.route`, predictions work and scores display N/A.

Check `http://127.0.0.1:8000/health`, `/trips`, and `/docs`. The trip list is discovered from the configured input directory; it does not contain hardcoded evaluator IDs.

## Teammate 4 - Laptop 2: UI, maps and report

Laptop 2 needs only Python and the `ui/` directory for the live evaluator interface:

```powershell
python -m http.server 8501 --bind 0.0.0.0 --directory ui
```

Open `http://localhost:8501` on Laptop 2. Enter `http://LAPTOP_1_IP:8000` into the API address field, click **Connect & load trips**, choose a trip and click **Match route**. Do not put `localhost:8000` on Laptop 2: that would refer to Laptop 2 itself. Laptop 1's IPv4 address can be obtained with `ipconfig` (Windows) or `hostname -I` (Linux).

Both laptops must use the same reachable network. If a campus network isolates clients, use an approved common hotspot or the SSH tunnel below. Keep the API on the trusted demo network. Permit the specific Python server/port through your firewall if prompted; do not disable the firewall.

The UI uses local SVG road geometry, so its route view needs no external tiles or CDN. The offline development Folium HTML uses Leaflet/CDNs and map tiles; internet is helpful for that separate development artifact.

## SSH transfer later

Replace `USER@HOST` with each teammate's actual SSH alias/address when supplied. Do not paste passwords or private keys into the chat.

```powershell
scp packages/teammate_3_model_api.zip USER@LAPTOP1:~/
scp packages/teammate_4_ui_report.zip USER@LAPTOP2:~/
```

On each Linux destination, extract its archive using `unzip`, enter `PathSnap`, then follow its `TEAMMATE_README.md`. On Windows, use `Expand-Archive` and run the PowerShell equivalents above.

An optional tunnel run from Laptop 2:

```bash
ssh -L 8000:127.0.0.1:8000 USER@LAPTOP1
```

Keep the tunnel running, then use `http://127.0.0.1:8000` in Laptop 2's UI. The request still travels to the API on Laptop 1. No remote SSH commands have been run yet.

## Final rehearsal and evidence

1. Confirm `/health` shows the expected model SHA-256, mode and ten input trips.
2. Run all ten supplied development trips through the API; select more than one in the UI.
3. Verify API logs appear on Laptop 1 when Laptop 2 sends requests.
4. Capture actual Laptop 1 server-console and Laptop 2 UI screenshots. Local same-computer screenshots are development evidence, not proof of a two-laptop deployment.
5. Put the genuine network screenshots into the consolidated report and regenerate it.
6. Review repository contents for credentials; commit source, configurations, selected model and results. Public GitHub publishing and the teacher's Google Form submission require the actual destination/account details and are not yet performed.
7. Submit by 3:00 PM IST. The teacher's form closes at 4:30 PM.
