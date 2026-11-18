# PathSnap

PathSnap matches GPS observations to directed road segments. Laptop 1 builds six features for each road within 75 metres, scores the candidates with LightGBM, and decodes a route with Viterbi. Laptop 2 lists trips and displays GPS points, the decoded route, accuracy when references are available, continuity, and inference time.

## Get the demo revision on both laptops

Run from the existing checkout in PowerShell. If checkout reports conflicting local edits, keep those edits and use a fresh clone for the demo.

```powershell
cd "$HOME\pathsnap-team-09"
git fetch origin
git switch --detach origin/fix/evaluator-integration
git rev-parse --short HEAD
if (-not (Test-Path .\.venv\Scripts\python.exe)) { py -3.12 -m venv .venv }
$Py = ".\.venv\Scripts\python.exe"
& $Py --version
```

Use Python 3.12 on each laptop. Both should print the same Git revision.

## Laptop 1 · API and model

Run this block once with the extracted **student** folder. Set the first line to the local path if the folder is elsewhere. Preparation reads the 80 training trips and the ten labelled student-test trips. The export fits the model using the training partition only. The separate reference export supplies scoring labels for the ten student-test trips.

```powershell
$Data = "$HOME\Downloads\Student folder\Student folder"
$Cache = ".\processed\demo-75m"
if (!(Test-Path "$Data\train") -or !(Test-Path "$Data\student_test")) { throw "Check the student data path." }
& $Py -m pip install -r PathSnap_Person4_requirements.txt
if ($LASTEXITCODE) { throw "Backend installation failed." }
& $Py -m model_experiments.checkpoint prepare "$Data" "$Cache" --radius 75
if ($LASTEXITCODE) { throw "Preparation failed." }
& $Py -m scripts.export_person4_model "$Cache" .\artifacts\demo --model lightgbm --seed 42
if ($LASTEXITCODE) { throw "Model export failed." }
& $Py -m scripts.student_references "$Cache" .\reference_labels\student_test
if ($LASTEXITCODE) { throw "Reference export failed." }
```

Start the API in the repo root. Leave the terminal running. The environment variables are per terminal.

```powershell
$env:PATHSNAP_DATA_ROOT = "$HOME\Downloads\Student folder\Student folder\student_test"
$env:PATHSNAP_ARTIFACTS_DIR = (Resolve-Path .\artifacts\demo).Path
$env:PATHSNAP_REFERENCE_LABELS_DIR = (Resolve-Path .\reference_labels\student_test).Path
& .\.venv\Scripts\python.exe -m uvicorn pathsnap_person4.api:app --host 0.0.0.0 --port 8000
```

In a second terminal, check readiness and find the Wi-Fi IPv4 address.

```powershell
Invoke-RestMethod http://127.0.0.1:8000/health
Invoke-RestMethod http://127.0.0.1:8000/trips
ipconfig
```

Expect `ready` and ten trips. The API prints live requests and appends prediction events to `logs/requests.jsonl`. Point references stay on the server and are read after prediction. When starting a separately authorised unlabelled demo, clear the student-test scoring directory from that server session using `Remove-Item Env:PATHSNAP_REFERENCE_LABELS_DIR -ErrorAction SilentlyContinue` before starting the server.

## Laptop 2 · evaluator UI

Join the same Wi-Fi or hotspot. Substitute the current Laptop 1 address if it differs from the address below. The API runs on port 8000 and the local browser UI on port 8501.

```powershell
$env:PATHSNAP_API_URL = "http://10.0.15.137:8000"
Invoke-RestMethod "$env:PATHSNAP_API_URL/health" -TimeoutSec 10
& .\.venv\Scripts\python.exe -m pip install -r ui/requirements.txt
if ($LASTEXITCODE) { throw "UI installation failed." }
& .\.venv\Scripts\python.exe -m ui.app
```

Open [the evaluator UI](http://127.0.0.1:8501), select a trip and click **Show route**. OpenStreetMap tiles and Folium's browser assets require internet access. If the API works on Laptop 1 but Laptop 2 cannot reach it, check the Wi-Fi IPv4 address and `Test-NetConnection 10.0.15.137 -Port 8000` on Laptop 2. Use a private network that permits device-to-device traffic and allow the API through the private-network firewall on Laptop 1.

## Verify all ten trips

With both services running, open another PowerShell terminal on Laptop 2 in the repository root. This checks each prediction endpoint and submits each trip through the evaluator form, so there are two predictions per trip. The CSV records per-trip classifier/HMM accuracy, continuity, correct adjacent pairs, counts of observations helped or hurt by decoding, and server inference time.

```powershell
& .\.venv\Scripts\python.exe -m ui.check_demo http://10.0.15.137:8000 --output .\logs\student_demo.csv
```

For the automated regression tests, install both laptops' dependencies into one environment and run:

```powershell
& .\.venv\Scripts\python.exe -m pip install -r PathSnap_Person4_requirements.txt -r ui/requirements.txt
& .\.venv\Scripts\python.exe -m pytest tests_person4 -q
```

## Experiment status

- Data cleaning, radius analysis, six features, model training, MLP loss curves, saved metrics and EDA maps are in the repository. The experiment guide documents how to run the classifiers together. [Model experiments](model_experiments/README.md)
- The live demo uses the seed-42 LightGBM export and the default decoder settings. It can run without scoring labels. The live map preserves route breaks and keeps reference arc IDs off Laptop 2. Labelled EDA maps are in `reports/maps/`.
- The existing RF/KNN comparison used two folds. The saved LightGBM/MLP comparison used five. A controlled five-model report run, Optuna tuning, decoder tuning, final report and a physical two-laptop rehearsal remain separate deliverables.

AI assistance was used to review the integration, implement fixes and regression tests, and draft the setup instructions.
