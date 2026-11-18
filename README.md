# PathSnap: map-matching and route reconstruction

PathSnap takes a noisy GPS trajectory and a road network, and reconstructs the
road segments (arcs) the vehicle actually drove on. It does this in two stages:

1. **Candidate classifier.** Every GPS point gets a small set of nearby
   candidate arcs. A supervised classifier scores each (point, arc) pair with the
   probability that the vehicle was on that arc.
2. **HMM + Viterbi.** The classifier probabilities are the emission scores of a
   hidden Markov model whose transitions use road-network connectivity. Viterbi
   decoding then picks the most likely connected sequence of arcs.

The system runs on two laptops: a FastAPI backend that serves the model (Laptop 1)
and a browser UI for the evaluator (Laptop 2).

## Repository layout

```
pathsnap/                  core library
  config.py                paths and default parameters
  io.py                    reading .track/.nodes/.arcs/.route files, track cleaning
  geo.py                   local metric projection, point-to-segment geometry
  network.py               road graph, spatial candidate search, bounded Dijkstra
  labels.py                aligns the reference route to GPS points (training labels)
  features.py              candidate generation + feature engineering
  models.py                the model zoo + Bayesian search spaces
  experiment.py            group-aware CV / student-test evaluation helpers
  metrics.py               classification + observation/route metrics
  hmm.py                   HMM / Viterbi decoder, route building, route metrics
  pipeline.py              end-to-end inference (used by API and scripts)
scripts/
  01_eda.py                dataset inventory, statistics, plots, Folium map
  02_build_dataset.py      candidate/feature tables for train and student_test
  03_compare_models.py     trains and compares 10 models (5-fold GroupKFold + student test)
  04_bayes_opt.py          Optuna TPE Bayesian optimisation of the chosen model
  05_export_model.py       fits the final model, writes artifacts/
  06_tune_hmm.py           decoder tuning + classifier-only vs HMM comparison
  07_maps.py               Folium maps of predictions vs reference
  08_fit_diagnostics.py    in-sample vs CV vs student-test scores (overfitting check), confusion grid
  09_concept_figures.py    explanatory figures from real data (nodes/arcs, candidates, transition, trellis)
api/main.py                FastAPI backend (Laptop 1)
ui/                        evaluator web UI (Laptop 2): index.html, app.js, style.css
artifacts/                 model.joblib + pipeline_config.json (used by the API)
results/                   all CSVs, JSON logs, plots and maps used in the report
explore.py, eda.py, clean_data.py   early exploration scripts
```

## Setup

Python 3.11+ is required.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

The raw data is not in git. Put it in `data/` (or point `PATHSNAP_DATA_DIR`
somewhere else):

```
data/
  train/<trip_id>/<trip_id>.track|.nodes|.arcs|.route
  student_test/<trip_id>/...
  evaluator_test/<trip_id>/...      (API laptop only, for the final evaluation)
```

On macOS, LightGBM and XGBoost need OpenMP: `brew install libomp`.

## Reproducing the experiments

Run these from the repository root, in order:

```bash
python scripts/01_eda.py              # results/eda/
python scripts/02_build_dataset.py    # cache/*.parquet, results/dataset/
python scripts/03_compare_models.py   # results/models/
python scripts/04_bayes_opt.py xgboost  # results/bayes_opt/ (with no argument it would tune the top CV Top-1 model, random forest)
python scripts/05_export_model.py     # artifacts/model.joblib, artifacts/pipeline_config.json
python scripts/06_tune_hmm.py         # results/hmm/, writes decoder params into the config
python scripts/07_maps.py             # results/maps/*.html
python scripts/08_fit_diagnostics.py  # results/models/train_vs_cv.csv + plots
python scripts/09_concept_figures.py  # results/concepts/*.png
```

`results/screenshots/` holds screenshots of the running system (UI and API log)
used in the report.

Random seeds are fixed in `pathsnap/config.py` (`SEED = 42`).

## Deployment (two laptops)

Both laptops must be on the same network.

**Laptop 1: API**

```bash
source .venv/bin/activate
uvicorn api.main:app --host 0.0.0.0 --port 8000
```

The API serves the trips in `data/evaluator_test/` by default. For a dry run
with the student-testing trips, set `PATHSNAP_EVAL_PARTITION=student_test`.
Every request is logged in this terminal: trip ID, number of points, classifier
and HMM accuracy, route F1 and timings. Find this laptop's IP with
`ipconfig getifaddr en0` (macOS) or `ipconfig` (Windows), and allow port 8000
through the firewall.

**Laptop 2: UI.** Only Python's standard library is needed:

```bash
python3 -m http.server 8080 --directory ui
```

Open `http://localhost:8080/?api=http://<LAPTOP1_IP>:8000`, or type the API
address in the panel and press Connect. The UI lists the evaluation trips. Pick
one, and the UI sends its ID to `POST /predict` and draws the result. The map
tiles and Leaflet load from the internet.

### API

| Method | Path | Body | Returns |
|---|---|---|---|
| GET | `/health` | | model name, decoder params, partition |
| GET | `/trips` | | trip IDs available for evaluation |
| POST | `/predict` | `{"trip_id": "00000000"}` | GPS points, predicted route polylines, classifier-only route, nearby roads, server-side scores, timings |

### Keeping the labels out of prediction

- `pathsnap.pipeline.PathSnap.predict` reads only `.track`, `.nodes` and
  `.arcs`. The `.route` file is opened only by `pathsnap.pipeline.evaluate`,
  after the prediction is finished.
- The API returns aggregate scores only. Reference-route geometry is never sent
  to the UI.
- Models, hyperparameters and decoder parameters are chosen using the 80
  training trips (group-aware CV, split by trip) and the 10 student-testing
  trips only. The evaluator-testing trips are never used for training or
  selection.

## Method summary

- **Projection.** Each trip is projected onto a local equirectangular plane in
  metres, centred on the trip. All trips are city-scale, so the distortion is
  negligible compared with GPS noise.
- **Labels.** `.route` is an ordered arc list, not one arc per point. A monotone
  dynamic-programming alignment assigns each GPS point to a route arc,
  minimising total distance. Points more than 100 m from their aligned arc are
  marked unreliable: they are excluded from training and from scoring.
- **Candidates.** All arcs within 60 m of a point (at most 12, at least 3) are
  kept. Both directions of a two-way street are separate candidates.
- **Features (23).** Perpendicular distance and its rank and gap to the best
  candidate; mean distance of neighbouring points to the arc; heading difference
  between GPS motion and arc direction; speed; arc length; projection position;
  candidate count; time gaps; GPS jump size; whether the candidate connects to
  the previous or next point's nearest arcs; two-way flag; node degrees.
- **HMM.** Hidden states are candidate arcs. Emission = `power * log p_classifier`.
  Transition = `-|network distance - straight-line distance| / beta`, where the
  network distance comes from a bounded Dijkstra search on the directed graph.
  Unreachable transitions get a fixed low log-probability. The trip is split
  at time gaps longer than `gap_seconds`. Decoding uses Viterbi.
- **Metrics.** Candidate-level accuracy, precision, recall, macro F1, log loss
  and confusion matrix. Observation-level Top-1 accuracy, candidate miss rate,
  route continuity, length-weighted route precision/recall/F1 and route
  mismatch fraction.

## Results

**Model comparison** (5-fold GroupKFold on the 80 training trips, then refit
and evaluated on the 10 student-testing trips). Full table:
`results/models/comparison.csv`.

| Model | CV macro F1 | CV Top-1 | Student-test Top-1 | Student-test log loss |
|---|---|---|---|---|
| Random forest | 0.9275 | 0.8774 | 0.8922 | 0.0659 |
| XGBoost | 0.9248 | 0.8764 | 0.8983 | 0.0632 |
| Extra trees | 0.9263 | 0.8746 | 0.8877 | 0.0699 |
| LightGBM | 0.9256 | 0.8743 | 0.8900 | 0.0755 |
| HistGradientBoosting | 0.9184 | 0.8732 | 0.8873 | 0.0700 |
| MLP | 0.9112 | 0.8710 | 0.8874 | 0.0681 |
| Decision tree | 0.9122 | 0.8698 | 0.8808 | 0.0744 |
| Logistic regression | 0.9214 | 0.8687 | 0.8810 | 0.0846 |
| KNN | 0.9153 | 0.8611 | 0.8760 | 0.1171 |
| Gaussian NB | 0.8009 | 0.8513 | 0.8689 | 0.8569 |

**Final model: XGBoost**, tuned with Optuna TPE (30 trials, 3-fold group CV,
objective = CV Top-1). Tuning raised 5-fold CV Top-1 from 0.8764 to 0.8785.
On student_test, however, Top-1 fell from 0.8983 to 0.8890. See
`results/bayes_opt/`.

**HMM decoder** (β = 5 m, emission power 0.5, top-8 states, gap split at 180 s;
tuned on out-of-fold training trips). See `results/hmm/`.

| | Point accuracy | Continuity | Route F1 | Route mismatch |
|---|---|---|---|---|
| Student-test, classifier only | 0.905 | 0.956 | 0.864 | 0.329 |
| Student-test, classifier + HMM | **0.947** | **0.979** | **0.989** | **0.023** |
| Train (OOF), classifier only | 0.868 | 0.954 | 0.832 | 0.484 |
| Train (OOF), classifier + HMM | **0.917** | **0.977** | **0.967** | **0.077** |

The HMM improved all 10 student-testing trips and 38 of the 40 training trips
used for tuning. Folium maps for every student-testing trip are in
`results/maps/`.

## Distributed Deployment (Laptop 1 & Laptop 2)

The system is designed for two-laptop evaluation across a local network (Wi-Fi or mobile hotspot):

### Laptop 1: Backend API Service
1. Bind to all interfaces on port `8000`:
   ```bash
   uvicorn api.main:app --host 0.0.0.0 --port 8000
   ```
2. Find Laptop 1's local IP address (`ipconfig getifaddr en0` on macOS).
3. The API loads the trained XGBoost model and configuration into memory on startup and serves `/health`, `/trips`, and `/predict`.

### Laptop 2: Evaluator UI Dashboard
1. Open `ui/index.html` in any web browser.
2. In the "API address" input box, enter `http://<Laptop-1-IP>:8000` and click **Connect**.
3. Select any evaluation trip from the sidebar to visualize raw GPS observations, candidate corridors, and the snapped HMM route with real-time metrics.
