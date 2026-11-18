# Person 4 — Chetan (PES2UG24CS131): integration instructions

This archive combines the latest team code supplied in `Pathsnap-team-09-main (4).zip` with
Person 4's HMM/Viterbi and FastAPI modules. **Review and run all code yourself before committing.**

## What is ready vs missing

Team repository currently has data preparation, 75 m candidate generation, features,
LightGBM/MLP scripts, saved evaluation CSVs and Folium maps. It does **not** contain a
trained `model.joblib`, a finished FastAPI deployment, or saved HMM-vs-classifier results.
The new `pathsnap_person4/` directory implements the decoder and API; `inference_adapter.py`
provides the real label-free candidate/feature interface to the current team functions.
`scripts/export_person4_model.py` retrains the team's classifier on **training** cache only.

## Windows commands (VS Code PowerShell terminal from repository root)

Use Python 3.12, recommended by your teammates; avoid Python 3.9.

```powershell
python --version
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pip install -r model_experiments/requirements.txt
.\.venv\Scripts\python.exe -m pip install -r PathSnap_Person4_requirements.txt
.\.venv\Scripts\python.exe -m pytest tests_person4 -q
```

Extract `Student folder.zip` so `Student folder/train` and `Student folder/student_test`
are accessible. Prepare the candidate/feature cache and train/export a real model (this
may take time; use a separate small training cache only for initial smoke testing):

```powershell
$Data = "$env:USERPROFILE\Desktop\Student folder"
.\.venv\Scripts\python.exe -m model_experiments.checkpoint prepare "$Data" processed/eval1-seed42 --radius 75
.\.venv\Scripts\python.exe -m scripts.export_person4_model processed/eval1-seed42 artifacts --model lightgbm
$env:PATHSNAP_DATA_ROOT = "$Data\student_test"
$env:PATHSNAP_ARTIFACTS_DIR = "artifacts"
$env:PATHSNAP_ADAPTER = "inference_adapter:build_candidates"
.\.venv\Scripts\python.exe -m uvicorn pathsnap_person4.api:app --host 0.0.0.0 --port 8000
```

From browser on laptop 1: `http://127.0.0.1:8000/docs`; first test `/health`, then `/trips`.
Then select a real trip ID from `/trips` and test `/predict/{trip_id}`.
The predicted route is in `predicted_route_lines` and details are in `observations`.
Optionally, only for local labeled student testing, set `$env:PATHSNAP_INCLUDE_FOLIUM='1'`
before launching; this invokes Person 1's server-side reference map function and returns
`map_html` for Person 3. This can produce a large response and is NOT suitable for a
hidden test that lacks `.route`; consult team about map confidentiality.

For laptop 2, use `ipconfig` to obtain laptop 1's IPv4 address. Connect both laptops
on the same Wi-Fi. Laptop 2 calls `http://<laptop-1-ip>:8000/trips` and
`http://<laptop-1-ip>:8000/predict/<trip_id>`. Allow Python/port 8000 through Windows
Firewall for a private network. Never expose the service to the public Internet.

## Caveats to check before submission

- Existing `model_experiments/checkpoint` trains and evaluates but **does not export models**.
  Use the new export command above and report its seed/training set.
- HMM decoding on tens of thousands of candidate rows can be expensive; measure API
  latency on all 10 student trips before the live demo.
- HMM tuning and real classifier-only vs HMM accuracy numbers are NOT established by
  this starter. Export/verify reference point labels separately, tune only on permitted
  data, and save results; do not claim an improvement without evidence.
- Current Folium map renders ordered selected arcs, but may visually bridge skipped
  or disconnected parts; ideally hand Person 1 the route polyline parts from
  `predicted_route_lines` for accurate gap-aware maps.
- Run the tests and verify integration on your own Windows laptop; no hardcoded output
  or test-label access in prediction paths.
