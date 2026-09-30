# Maré Hotels' cancellation model — tune, register, serve, package

## What we're trying to do

**Take a tracked model to a live endpoint, so the next, better model can replace it without anyone changing code. Then package the training job, so anyone can run it.**

- **Where it starts:** `train.py` trains Maré Hotels' booking-cancellation model, an XGBoost classifier, and tracks each run in MLflow. Today's model scores 0.9017 test ROC-AUC. It exists as a run and a logged model (`models:/m-…`), with no name, no version, and no record that it's the one in use.
- **Where it ends:** a tuned challenger (0.9169), both models as versions of `booking-cancellation` in MLflow's Model Registry, two HTTP endpoints answering from whatever `@champion` points at, and the challenger promoted by moving that one alias. Then `train.py` becomes an MLflow Project, and one command rebuilds the champion from its own run: 0.9169 again.
- **What it offers:**

| Route | Served by | Called by | Answers |
|---|---|---|---|
| `POST /invocations` | `mlflow models serve` | services and batch jobs that already hold the model's 20 columns | `{"predictions": [[P(kept), P(cancelled)]]}` |
| `GET /ping` | `mlflow models serve` | a load balancer | 200 once the model is loaded |
| `POST /predict` | `app.py` | the booking engine | `p_cancel`, `reconfirm`, `threshold`, `model_version` |
| `GET /health` | `app.py` | monitoring | which model version is answering |

| Command | Runs | Used for |
|---|---|---|
| `mlflow run . -e train -P max_depth=8` | `train.py` with those parameters; the defaults are today's model | trying a configuration |
| `mlflow run . -e retrain -P config=runs:/<run_id>/config.json` | `train.py --config` with a configuration a run logged | rebuilding a model from its own run |

## How we're doing it

**One notebook, six steps; each fixes something the step before it can't do.**

| Part | The problem | The fix |
|---|---|---|
| P1 | Today's model is known only by an ID | nothing yet: it sets the bar, 0.9017 test ROC-AUC |
| P2 | A search tried by hand leaves no record, and its winner gets retyped | Optuna as one parent run with 20 child runs; `train.py --config runs:/…/best_config.json` trains the winner |
| P3 | "Which model is in use" isn't written down anywhere | the Model Registry: versions 1 and 2 of `booking-cancellation`, aliases `@champion` and `@challenger` |
| P4 | Callers need an endpoint, and a model signature checks names and types, not values | `mlflow models serve` for model-shaped callers; `app.py` with a value contract for the booking engine |
| P5 | Replacing the model would mean changing every consumer | a gate in code, one alias move, a restart; rollback is the same move back |
| P6 | Running `train.py` takes knowing its flags, its environment and where its runs go | an MLflow Project: `MLproject` names two commands, `python_env.yaml` their environment, and `mlflow run . -e retrain` rebuilds the champion from its run's `config.json` |

- Every step runs in the notebook. Both endpoints are called in-process (FastAPI's `TestClient`), so no server has to be running.
- The terminals are for seeing it live: the MLflow UI, and optionally the two real servers in P4–P5.
- P6's `mlflow run` lines start `train.py` from the notebook, in the notebook's own environment. Building a fresh environment from `python_env.yaml` is a terminal step.

## How to go through this folder

| Parts | What has to be running |
|---|---|
| P0 – P3 | the notebook; `mlflow ui` in a terminal for the "In the UI" steps (optional: every result is also printed in the notebook) |
| P4 – P5 | the notebook; optionally `mlflow models serve` and `uvicorn app:app` in two more terminals |
| P6 | the notebook; optionally one terminal for `mlflow run --env-manager uv` |

1. Use a Python 3.12 environment and install the pinned versions (the notebook's first cell runs the same command):

   ```bash
   pip install -r requirements.txt
   ```

   On a Mac, XGBoost needs the OpenMP runtime, once: `brew install libomp`.

2. Open `v2_MLflow_Tuning_Registry_Serving.ipynb` with that environment as the kernel. Run P0 – P1.
3. Start the UI in a terminal, from this folder, in the same environment:

   ```bash
   cd "<this folder>"
   mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5001
   ```

   Open http://127.0.0.1:5001 and switch the toggle at the top-left from **GenAI** to **Model training**.
4. Run P2 (the search takes about 30 seconds; the trials appear under `tune-xgb` in the UI as they finish) and P3 (then open **Model registry** in the UI).
5. At P4, to see the endpoints live, start them in two more terminals from this folder:

   ```bash
   MLFLOW_TRACKING_URI=sqlite:///mlflow.db mlflow models serve \
     -m "models:/booking-cancellation@champion" --env-manager local --port 5002
   ```

   ```bash
   uvicorn app:app --port 8001
   ```

   Try http://127.0.0.1:8001/docs → **POST /predict** → **Try it out**, or the `curl` line in P4.
6. After P5's promotion cell, restart both servers (Ctrl+C, then the same command): `/health` reports version 2.
7. Run P6. Its `mlflow run` lines use the notebook's environment (`--env-manager local`). To have MLflow build the environment from `python_env.yaml` instead, in a terminal from this folder:

   ```bash
   pip install uv        # once: MLflow calls uv to build the environment
   MLFLOW_TRACKING_URI=sqlite:///mlflow.db mlflow run . -e train \
     --experiment-name booking-cancellations --env-manager uv
   ```

   The first run takes a minute or two and puts about 600 MB under `~/.mlflow/envs/`; later runs reuse it.
8. Stop every server with Ctrl+C when you're done. To start again from an empty store, delete `mlflow.db` and `mlruns/`.

### Files, by the step that needs them

| Needed at | File | What it is |
|---|---|---|
| every step | `v2_MLflow_Tuning_Registry_Serving.ipynb` | the notebook |
| P0 | `requirements.txt` | pinned versions (MLflow 3.16.1, XGBoost 3.2.0, Optuna 4.9.0, FastAPI 0.139.0, scikit-learn 1.7.2) |
| P1 – P2 | `train.py` | the training job: trains, scores on the test set, logs the run and the model |
| P1 – P2, and `train.py` | `data/bookings_2015_2017.csv.gz` | 118,300 bookings with a known outcome |
| P4 – P5 | `data/upcoming_arrivals.csv` | 1,090 bookings arriving next week, outcome unknown |
| P4 – P5 | `app.py` | the booking engine's service: the same code as the notebook's P4 cells |
| P6 | `MLproject` | the project's two commands, `train` and `retrain`, and their parameters |
| P6 | `python_env.yaml` | the environment those commands need: Python 3.12 and `requirements.txt` |
| created by running | `mlflow.db`, `mlruns/` | the tracking store and registry: run metadata, artifacts, models |
| not needed | `v1_MLflow_Tuning_Registry_Serving.ipynb` | the notebook as it was before P6: P0 – P5 and the wrap-up |
| not needed | `_build_nb.py`, `update.md` | how the notebook is built, and its change log |

---

## Reference

- **Ports:** 5001 the MLflow UI · 5002 `mlflow models serve` · 8001 `app.py`. Not 5000: macOS keeps it for AirPlay Receiver.
- **Tracking URI:** `sqlite:///mlflow.db`, relative to the folder a process starts in. `train.py` and `app.py` read `MLFLOW_TRACKING_URI` first.
- **Experiment:** `booking-cancellations`. Run tags: `stage = pipeline | tuning | trial`; on project runs also `mlflow.source.type = PROJECT` and `mlflow.project.entryPoint = train | retrain`.
- **Registered model:** `booking-cancellation`. Aliases: `@champion` (what the endpoints load), `@challenger` (a candidate, until promoted).
- **`train.py`:** `python train.py` trains today's configuration (XGBoost: 300 trees, depth 6, learning rate 0.1); `--config runs:/<run_id>/best_config.json` trains the one a tuning run logged.
- **`MLproject`:** entry point `train` takes `learning_rate`, `max_depth`, `min_child_weight` and `n_estimators` (the defaults are today's configuration); `retrain` takes `config`, a `runs:/…` address. In a terminal, `mlflow run` needs `MLFLOW_TRACKING_URI` and `--experiment-name booking-cancellations`; in the notebook the kernel's environment already names both.
- **Data:** Hotel Booking Demand dataset — Antonio, de Almeida & Nunes, *Data in Brief* 22 (2019), CC BY 4.0.
