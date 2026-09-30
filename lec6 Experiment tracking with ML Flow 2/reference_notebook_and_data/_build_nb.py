"""Build the lec 6 notebook: tune, register, serve, promote, then package (or, with Airflow, automate).

This script is the master copy, and the same file sits in both folders. The folder decides what
it builds: one with a `dags/` directory gets the Airflow Part (P6: automate) and its notebook
name; one without gets the MLflow Projects Part (P6: package).

    python _build_nb.py            -> writes the notebook next to this file
    python _build_nb.py --out DIR  -> writes it to DIR instead (for diffing first)

train.py, app.py, requirements.txt (and MLproject + python_env.yaml, or dags/weekly_retrain.py)
are real files beside this one, hand-maintained. The notebook defines app.py's code itself (P4),
and the DAG's (P6); it quotes MLproject and python_env.yaml straight from the files. Every build
compares the code with those files, and MLproject's commands with train.py's flags, and says
whether they still match.

Every number the prose quotes from an executed run lives in MEASURED below, so a re-run with
different results means updating one table, then reading the prose that uses it.

Nothing in the notebook starts a server. `mlflow ui`, `mlflow models serve`, `uvicorn` and
`airflow standalone` are fenced commands for a terminal you own; the notebook reaches both apps
through TestClient, and runs the DAG with dag.test(), in-process, with no port.
"""

import argparse
import ast
import json
import re
from pathlib import Path

HERE = Path(__file__).parent
AIRFLOW = (HERE / "dags").is_dir()  # the Airflow folder has dags/; the plain one doesn't
# The plain folder's v1 (no P6) stays there as executed; this script builds v2, which adds P6: package
NB_NAME = ("v1_MLflow_Tuning_Registry_Serving_Airflow.ipynb" if AIRFLOW
           else "v2_MLflow_Tuning_Registry_Serving.ipynb")
KERNELSPEC = ({"display_name": "airflow3.12", "language": "python", "name": "airflow3.12"} if AIRFLOW
              else {"display_name": "dev3.12", "language": "python", "name": "python3"})

# Numbers the prose quotes, as the executed notebook printed them. ⟪name⟫ in any cell is replaced
# by MEASURED[name]; the build refuses to finish if one is missing.
MEASURED = {
    # P1: train.py defaults (xgb-depth6)
    "p1_test": "0.9017",
    "p1_size": "0.9",
    # P2: the search and the challenger (xgb-tuned)
    "p2_seconds": "30",
    "p2_random_best": "0.9114",
    "p2_best_trial": "trial-13",
    "p2_best_val": "0.9134",
    "p2_best_lr": "0.089",
    "p2_best_depth": "12",
    "p2_best_mcw": "1.03",
    "p2_tuned_test": "0.9169",
    "p2_tuned_size": "5.6",
    "p2_pattern_title": "The best trials grow deep trees that split on little evidence.",
    "p2_pattern_1": "The top four are 12 to 14 levels deep, with `min_child_weight` under 3 and `learning_rate` between 0.07 and 0.13. Today's model stops at depth 6.",
    "p2_pattern_2": "The first 10 trials are random. TPE's proposals after them found the top three: trial-13, trial-16 and trial-14. The best random trial, trial-08, is fourth.",
    "p2_try_question": "Find the trials whose model is under 2 MB and still scored above 0.907 on validation.",
    "p2_try_filter": "and metrics.model_size_mb < 2 and metrics.val_roc_auc > 0.907",
    "p2_try_answer": "Two qualify, both 8 levels deep: `trial-18` (1.8 MB, 0.9093) and `trial-15` (1.5 MB, 0.9077). They're within 0.006 of the best trial at about a third of its size.",
    # P4: BK218304 through version 1
    "p4_as_sent": "0.7255",
    "p4_as_sent_2dp": "0.73",
    "p4_ota": "0.3773",
    "p4_minus1": "0.2004",
    "p4_city": "0.7771",
    "p4_flip_count": "two",
    "p4_ota_line": "**200** · `'Online Travel Agent'`: 0.73 → 0.38. The encoder has never seen that code, so every `market_segment` column is 0. The model answers for a booking with no sales channel, and the reconfirm call flips.",
    "p4_minus1_line": "**200** · `lead_time = -1`: 0.73 → 0.20, another flip. The trees file −1 with same-day bookings (`lead_time = 0` scores exactly the same), and only 7% of those cancel.",
    "p4_city_line": "**200** · `'city hotel'`: 0.73 → 0.78. A small move this time, but still a guess about a hotel the model doesn't know.",
    # P5: the gate, the flips, the promotion
    "p5_gain": "+0.0152",
    "p5_flips": "77",
    "p5_after": "0.7908",
    "p5_flips_line": "About the same number of reconfirm calls (268 → 271), but 77 bookings trade places: 37 leave the list and 40 join it.",
    # P6 in the plain folder: the two project runs, and the environment uv builds (measured in a terminal)
    "pk_depth8_test": "0.9092",
    "pk_rebuilt_test": "0.9169",
    "pk_uv_line": "The first run takes a minute or two: MLflow creates an environment under `~/.mlflow/envs/` (about 600 MB) and installs `requirements.txt` into it. Later runs find it there and start in seconds.",
    # P6 in the Airflow folder: the week's outcomes and the DAG's Friday run
    "p6_right_line": "**Not on this week.** Version 1 ranked the week's bookings slightly better (0.8372 against 0.8330), and was right on 41 of the 77 flipped decisions to version 2's 36.",
    "p6_noise_line": "On 1,090 bookings a gap that small is noise: one ROC-AUC moves about ±0.014 from sampling alone. One week can't overturn a test set of 23,660; many weeks can, which is why the DAG below logs this check every Friday.",
    "p6_oot_line": "What isn't noise: both versions score 0.06 to 0.08 below their test ROC-AUC. The test set is a random 20% of 2015–2017; the week after the data ends is harder, and only outcomes like these show by how much.",
    "p6_live_auc": "0.8330",
    "p6_gain": "-0.0001",
    "p6_promote_label": "skipped: @champion stays on version 2",
    "p6_run_line_1": "`check_live`: version 2 scored 0.8330 ROC-AUC on the week it served. Of its 271 reconfirm calls, 68% were bookings that did cancel, against 32% for the week as a whole.",
    "p6_run_line_2": "`retrain` → `register`: version 3 is the champion's configuration with the week's 1,090 bookings added to training. It scored 0.9167 on test, −0.0001 against version 2, so `passes_gate` kept the champion and `promote` was skipped. `@challenger` now names version 3.",
    "p6_think": "Version 2 passed P5's gate by +0.0152 on the test set, yet on the week that followed it ranked bookings no better than version 1, and both scored 0.06 to 0.08 lower than on test. What would you change in how the gate measures a challenger?",
    "p6_ui_steps": """- Open http://localhost:8080 and sign in as `admin` with the password it prints; it's also saved in `airflow/simple_auth_manager_passwords.json.generated`.
- **Dags** → `weekly_retrain`. The toggle beside its name is off: the DAG is paused. Leave it that way. Unpaused, the scheduler would start with the most recent Friday, which has no outcomes file.
- The grid on the left has a column per run and a row per task. The 1 September run is green down to `passes_gate`, with `promote` skipped (pink); the 8 September run is green for `find_outcomes` only. The icons above the grid switch to a graph of the six tasks (or press `g`).
- Click a square for that task's page. **Storage** → **XCom** shows what it returned: `retrain`'s is the `model_uri = …` line that `register` received. **Code** shows `dags/weekly_retrain.py`.
- **Logs** stays empty for these two runs: `dag.test()` printed each task's output in this notebook instead. Runs the scheduler starts keep a log per task.
- A run's page shows its **Logical Date** (1 or 8 September 2017, in your browser's time zone). Its start date says 2017 as well, and its duration years: `dag.test()` records a run as starting on its logical date.
- 8080 is Airflow's default port. Stop it with Ctrl+C.""",
}

cells = []


def md(text):
    cells.append(
        {
            "cell_type": "markdown",
            "id": f"md-{len(cells):03d}",
            "metadata": {},
            "source": text.strip("\n").splitlines(keepends=True),
        }
    )


def code(text):
    cells.append(
        {
            "cell_type": "code",
            "id": f"code-{len(cells):03d}",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": text.strip("\n").splitlines(keepends=True),
        }
    )


PARTS = [
    ("P0", "setup"),
    ("P1", "today's model"),
    ("P2", "tune"),
    ("P3", "register"),
    ("P4", "serve"),
    ("P5", "promote"),
    ("P6", "automate" if AIRFLOW else "package"),
    ("P7", "wrap-up"),
]
WRAP = PARTS[-1][0]  # the wrap-up's tag


def here(part, mission):
    """The 'you are here' strip: done stages ✓, the current one called out."""
    tags = [tag for tag, _ in PARTS]
    idx = tags.index(part)
    bits = []
    for i, (_, label) in enumerate(PARTS):
        if i < idx:
            bits.append(f"{label} ✓")
        elif i == idx:
            bits.append(f"[▶ {label.upper()}]")
        else:
            bits.append(label)
    lines, line = [], ""
    for i, bit in enumerate(bits):
        piece = bit if i == 0 else f" → {bit}"
        if line and len(line) + len(piece) > 92:
            lines.append(line)
            line, piece = "", f"→ {bit}"
        line += piece
    lines.append(line)
    strip = "```\n" + "\n".join(lines) + "\n```"
    return f"{strip}\n\n*You are here → **Part {idx} of {len(PARTS) - 1}**: {mission}.*"


def part(tag, title, mission, body, question=None):
    """A Part's opening cell: heading, an optional framing question, the 'you are here' strip, the body."""
    lead = f"{question}\n\n" if question else ""
    md(f"## {tag} · {title}\n\n{lead}{here(tag, mission)}\n\n{body.strip()}")


def boxed(lines, width):
    """A box whose right border is padded programmatically, so it never drifts."""
    top = "  ┌" + "─" * (width + 2) + "┐"
    bottom = "  └" + "─" * (width + 2) + "┘"
    rows = [f"  │ {line.ljust(width)} │" for line in lines]
    return "\n".join([top, *rows, bottom])


# every mermaid picture wraps node labels at 360 px instead of mermaid's default 200
MERMAID_HEAD = """```mermaid
---
config:
  flowchart:
    wrappingWidth: 360
---"""

PALETTE = """    classDef code fill:#e8f0fe,stroke:#4285f4,color:#174ea6
    classDef stored fill:#f3e8fd,stroke:#9334e6,color:#681da8
    classDef guard fill:#fce8e6,stroke:#d93025,color:#a50e0e
    classDef endpoint fill:#e6f4ea,stroke:#34a853,color:#137333
    classDef person fill:#fff3e0,stroke:#ef6c00,color:#8a3800"""


def picture(tag, title, point, graph, classes, bullets):
    """'Pn in one picture': heading, a bold line, the diagram, then 1–2 bullets."""
    diagram = f"{MERMAID_HEAD}\n{graph.strip()}\n{PALETTE}\n{classes.strip(chr(10))}\n```"
    md(f"### {tag} in one picture — {title}\n\n**{point}**\n\n{diagram}\n\n{bullets.strip()}")


# ---------------------------------------------------------------- title + map

_title = "MLflow and Airflow: tune, register, serve, automate" if AIRFLOW else "MLflow: tune, register, serve, package"
_ends = (
    "- **Where it ends:** a tuned challenger, both models as versions of `booking-cancellation` in the Model Registry, "
    "two endpoints answering from `@champion`, and the challenger promoted by moving that one alias."
    + (" Then an Airflow DAG runs that loop every Friday on the week's outcomes: check the champion, retrain, register, gate, promote."
       if AIRFLOW else
       " Then `train.py` becomes an MLflow Project, two named commands and a declared environment, and one of them rebuilds the champion from its own run.")
)
_rows = (
    "| P6 | Automate | Airflow: a DAG, `@task`, a schedule, `dag.test()` | `weekly_retrain`: every Friday, the week's outcomes checked, a retrain registered and gated |\n"
    "| P7 | Wrap-up | | the whole path on one page |"
    if AIRFLOW else
    "| P6 | Package | MLflow Projects: `MLproject` · entry points · `mlflow run` | runs `project-train` and `project-retrain`; the champion rebuilt, test ROC-AUC ⟪pk_rebuilt_test⟫ again |\n"
    "| P7 | Wrap-up | | the whole path on one page |"
)

md(
    f"""
# {_title} — Maré Hotels' cancellation model

**A tuned model goes through the Model Registry to a live endpoint, and the next model takes its place without a code change{"; then a scheduler runs that loop every week" if AIRFLOW else "; then the training job is packaged, so anyone can run it"}.**

- **The business:** Maré Hotels runs a city hotel in Lisbon and a resort in the Algarve. Every upcoming booking gets a P(cancel); the revenue team reconfirms the ones at 0.5 or above and overbooks against the rest.
- **Where it starts:** `train.py` trains the model, an XGBoost classifier, and tracks each run in MLflow. Today's model scores ⟪p1_test⟫ test ROC-AUC. It exists as a run and a logged model, but nothing records that it's the one in use, and replacing it means re-pointing everything that loads it.
{_ends}

| Part | Step | MLflow (and friends) you use | You end up with |
|---|---|---|---|
| P1 | Today's model | `train.py` · runs · logged models | run `xgb-depth6`: test ROC-AUC ⟪p1_test⟫ |
| P2 | Tune | Optuna · nested runs · `search_runs` · a config handed over as an artifact | 20 trials under `tune-xgb`; the challenger run `xgb-tuned` |
| P3 | Register | Model Registry: versions · aliases · descriptions · tags | `booking-cancellation` version 1 `@champion`, version 2 `@challenger` |
| P4 | Serve | `mlflow models serve` · signatures · FastAPI + pydantic | `POST /invocations`; `app.py` with `POST /predict` and `GET /health` |
| P5 | Promote | a gate in code · `set_registered_model_alias` | `@champion` → version 2, and both endpoints answering with it |
{_rows}
"""
)

md(
    r"""
### What gets built

**Two HTTP endpoints serve the same registered model; only one of them checks values.**

| Route | Served by | Called by | Answers | Built in |
|---|---|---|---|---|
| `POST /invocations` | `mlflow models serve` | services and batch jobs that already hold the model's 20 columns | `{"predictions": [[P(kept), P(cancelled)]]}` | P4 |
| `GET /ping` | `mlflow models serve` | a load balancer | 200 once the model is loaded | P4 |
| `POST /predict` | `app.py` (FastAPI) | the booking engine | `p_cancel`, `reconfirm`, `threshold`, `model_version` | P4 |
| `GET /health` | `app.py` | monitoring | which model version is answering | P4 |

One booking through `POST /predict`:

"""
    + MERMAID_HEAD
    + r"""
flowchart LR
    ENGINE(["booking engine<br/>POST /predict · BK218304"])
    CHECK{{"Booking schema<br/>codes the model knows · counts ≥ 0"}}
    BAD(["422<br/>names the field"])
    MODEL["model loaded at startup<br/>models:/booking-cancellation/N"]
    REG[("Model Registry<br/>@champion → version N")]
    OK(["200<br/>p_cancel · reconfirm · model_version N"])
    ENGINE --> CHECK
    CHECK -- "bad value" --> BAD
    CHECK -- "valid" --> MODEL
    REG -. "resolved once,<br/>at startup" .-> MODEL
    MODEL --> OK
"""
    + PALETTE
    + r"""
    class ENGINE,OK endpoint
    class CHECK,BAD guard
    class MODEL code
    class REG stored
```

- 🟩 the caller and the answer · 🟥 the value contract · 🟦 the model · 🟪 the registry.
- `mlflow models serve` has the same shape without the 🟥 step: its only check is the model's signature, column names and types (P4).
"""
    + (
        """- **Plus one scheduled job (P6):** `weekly_retrain`, an Airflow DAG that every Friday checks the champion on the week's outcomes, retrains, registers and gates, with no one running it.
"""
        if AIRFLOW else
        """- **Plus one packaged training job (P6):** `MLproject` names two commands, `train` and `retrain`, and `python_env.yaml` names their environment. `mlflow run . -e retrain` rebuilds a model from the `config.json` its run logged.
"""
    )
)

_box = boxed(
    [
        "END RESULT: every consumer loads models:/booking-cancellation@champion. A better",
        "model reaches them through a gate, one alias move and a restart; rolling back",
        "is the same move in reverse.",
        *(["Every Friday, an Airflow DAG runs that loop on the week's outcomes and records",
           "each run, whichever way the gate decides."] if AIRFLOW else
          ["The training job is an MLflow Project: `mlflow run` starts it by name, and one",
           "command rebuilds any version from the config its run logged."]),
    ],
    width=81,
)

_p6_map = (
    r"""  [P6] DAG weekly_retrain, Fridays 06:00: find_outcomes ─► check_live
                                          └─► retrain ─► register ─► passes_gate ─► promote
      │
      ▼
"""
    if AIRFLOW else
    r"""  [P6] MLproject (entry points train · retrain) + python_env.yaml ──► an MLflow Project
       !mlflow run . -e retrain -P config=runs:/<champion_run_id>/config.json
                                  ──► run project-retrain · test_roc_auc ⟪pk_rebuilt_test⟫ again
      │
      ▼
"""
)

md(
    r"""
## The map

```
         MARÉ HOTELS — from a tracked model to a served one you can replace

  data/bookings_2015_2017.csv.gz ··· 118,300 bookings with a known outcome (is_canceled)
      │
      ▼
  [P1] !python train.py --run-name xgb-depth6 ──► test_roc_auc ⟪p1_test⟫ · models:/m-…
      │                                   an ID only: no name, no version, no role
      ▼
  [P2] run "tune-xgb" ── 20 child runs, trial-00 … trial-19 ── best_config.json
       !python train.py --config runs:/<tuning_run_id>/best_config.json ──► xgb-tuned
      │
      ▼
  [P3] mlflow.register_model(...) ──► booking-cancellation   v1 @champion · v2 @challenger
      │
      ▼
  [P4] mlflow models serve -m models:/booking-cancellation@champion ──► POST /invocations
       app.py: Booking ─► load_champion() ─► POST /predict · GET /health
      │
      ▼
  [P5] gate: test ROC-AUC ⟪p5_gain⟫ ≥ 0.002 · ⟪p2_tuned_size⟫ MB < 50 MB ──► @champion → v2 ──► restart
      │
      ▼
"""
    + _p6_map
    + _box
    + r"""
```
"""
)

# ---------------------------------------------------------------- P0 : setup

part(
    "P0",
    "Setup",
    "install the pinned versions, import everything, point MLflow at the store",
    "**Everything this notebook needs sits beside it: `requirements.txt`, `data/`, `train.py`, `app.py`"
    + (", `dags/`" if AIRFLOW else ", `MLproject`, `python_env.yaml`")
    + ".**\n\n"
    + "- The terminal commands later (`mlflow ui`, `mlflow models serve`, `uvicorn`"
    + (", `airflow standalone`" if AIRFLOW else ", `mlflow run`")
    + ") run from this folder, in the environment this notebook's kernel uses.\n"
    + (
        "- **This folder gets its own virtual environment.** Airflow pins many of its dependencies "
        "(FastAPI below 0.137, for one), so `requirements.txt` goes into a fresh environment, not one "
        "shared with other projects.\n"
        if AIRFLOW else ""
    )
    + r"""- **On a Mac, XGBoost needs the OpenMP runtime:** `brew install libomp`, once, before the install cell.
- **What this Part builds:** the imports, one environment setting, and `client`, the MLflow client the registry calls go through.

| New word | What it means here |
|---|---|
| tracking URI | where MLflow reads and writes runs, models and the registry: `sqlite:///mlflow.db`, a file beside this notebook |
| `MlflowClient` | MLflow's lower-level API, on the same store; registry operations (aliases, version tags) live on it |
""",
)

code(
    r"""
# Install the pinned versions (a no-op when they're already there)
%pip install -q --disable-pip-version-check -r requirements.txt
"""
)

code(
    r"""
# Print how long every cell takes
%load_ext autotime
"""
)

code(
    r"""
# One environment setting first (this kernel and every `!python` line inherit it), then the imports
import os

os.environ["MLFLOW_ENABLE_ARTIFACTS_PROGRESS_BAR"] = "false"  # no download progress bars in the outputs

import json
import logging
import pickle
import time
from typing import Literal

import matplotlib.pyplot as plt
import mlflow
import optuna
import pandas as pd
import sklearn
import xgboost
from fastapi import FastAPI
from fastapi.testclient import TestClient
from mlflow import MlflowClient
from mlflow.pyfunc import scoring_server
from pydantic import BaseModel, Field
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

print(f"mlflow {mlflow.__version__} · optuna {optuna.__version__} · scikit-learn {sklearn.__version__} · "
      f"xgboost {xgboost.__version__} · pandas {pd.__version__}")
"""
)

code(
    r"""
# Point MLflow at the store beside this notebook, and select the experiment every run goes into
mlflow.set_tracking_uri("sqlite:///mlflow.db")               # also exported as MLFLOW_TRACKING_URI: `!python train.py` logs here too
experiment = mlflow.set_experiment("booking-cancellations")  # created on first use, selected after that
client = MlflowClient()                                      # same store; P3 and P5 go through it for the registry

print("tracking URI:", mlflow.get_tracking_uri())
print("experiment  :", experiment.name, "| id", experiment.experiment_id)
"""
)

md(
    r"""
### Open the UI — in a terminal

**The UI reads the same store: start it yourself and leave it running.**

```bash
cd "<the folder that contains this notebook>"
mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5001
```

- Open http://127.0.0.1:5001. If the sidebar lists GenAI pages, flip the toggle at the top-left from **GenAI** to **Model training**: the sidebar becomes **Runs · Models · Traces** plus **Model registry**.
- `--port 5001` because macOS keeps MLflow's default port, 5000, for AirPlay Receiver. Stop it with Ctrl+C.
"""
)

# ---------------------------------------------------------------- P1 : today's model

part(
    "P1",
    "Today's model",
    "meet the data, then train the model the team runs today",
    r"""
**`train.py` trains the cancellation model and tracks the run. What it trains with its default settings is today's model: the one to beat.**

- `train.py` is one tracked training job: read the data, fit, score on the test set, log it all, print the run ID and model URI.
- **What this Part builds:** the split every later Part uses (`X_train`, `X_val`, `X_test`), and the run `xgb-depth6` with its logged model.

| New word | What it means here |
|---|---|
| XGBoost | gradient-boosted decision trees: each new tree corrects the errors of the ones before it; `XGBClassifier` is its scikit-learn-style model |
| experiment | a named group of runs for one prediction problem: `booking-cancellations` |
| run | one execution of training code, with its params, metrics, tags and artifacts |
| logged model | a model saved in MLflow by `log_model`, with an ID of its own: `models:/m-…` |
| signature | the model's input columns and types and its output shape, checked on every generic call |
| `pyfunc_predict_fn` | which method a generic `predict()` calls; `predict_proba` here, so callers get probabilities |
| `run.outputs.model_outputs` | the logged models a run produced |
""",
)

picture(
    "P1",
    "a model with an ID, not a name",
    "One command trains today's model; what comes out is a run and a logged model known only by its ID.",
    r"""
flowchart LR
    DATA[("data/bookings_2015_2017.csv.gz<br/>118,300 bookings")]
    TRAIN["!python train.py<br/>--run-name xgb-depth6"]
    RUN[("run xgb-depth6<br/>test_roc_auc ⟪p1_test⟫ · ⟪p1_size⟫ MB")]
    LM[("logged model models:/m-…<br/>predict → P(kept), P(cancelled)")]
    ASK(["which model is in use?<br/>nothing records it"])
    DATA --> TRAIN --> RUN --> LM -.-> ASK
""",
    r"""
    class TRAIN code
    class DATA,RUN,LM stored
    class ASK person
""",
    r"""
- 🟦 code · 🟪 stored in MLflow · 🟧 a question a person has to answer.
- Everything a consumer needs to load the model is there (P3 loads it); what's missing is a name for "the one in use".
""",
)

code(
    r"""
# 118,300 past bookings with a known outcome: one row per booking
bookings = pd.read_csv("data/bookings_2015_2017.csv.gz")

print(f"{len(bookings):,} bookings · arrivals {bookings.arrival_date.min()} → {bookings.arrival_date.max()} · cancelled {bookings.is_canceled.mean():.1%}")
bookings.head(3)
"""
)

md(
    r"""
### What the columns mean

**Twenty model inputs, two identifiers, one target. The codes below are exactly what the model was trained on.**

| Column | Meaning |
|---|---|
| `booking_id`, `arrival_date` | identifiers — not model inputs |
| `hotel` | `City Hotel` (Lisbon) or `Resort Hotel` (Algarve) |
| `lead_time` | days between making the booking and the arrival date |
| `stays_in_weekend_nights`, `stays_in_week_nights` | length of stay, split into weekend and weekday nights |
| `adults`, `children`, `babies` | party size |
| `meal` | `BB` bed & breakfast · `HB` half board · `FB` full board · `SC` / `Undefined` no meal package |
| `market_segment` | how the booking was sold: `Online TA` (online travel agent), `Offline TA/TO` (travel agent / tour operator), `Groups`, `Direct`, `Corporate`, `Complementary`, `Aviation`, `Undefined` |
| `distribution_channel` | `TA/TO`, `Direct`, `Corporate`, `GDS` (global distribution system), `Undefined` |
| `is_repeated_guest` | `1` = the guest has stayed before |
| `previous_cancellations`, `previous_bookings_not_canceled` | the guest's booking history |
| `reserved_room_type` | room category, as an anonymised letter code (`A` … `H`, `L`, `P`) |
| `deposit_type` | `No Deposit` · `Non Refund` (paid in full, no refund) · `Refundable` |
| `customer_type` | `Transient` (individual) · `Transient-Party` (linked to another booking) · `Contract` · `Group` |
| `days_in_waiting_list` | days the booking waited before it was confirmed |
| `adr` | average daily rate, EUR |
| `required_car_parking_spaces`, `total_of_special_requests` | extras the guest asked for |
| `is_canceled` | **target** — `1` = cancelled |

- P4's service accepts these codes and nothing else.

*Source: Hotel Booking Demand dataset — Antonio, de Almeida & Nunes, Data in Brief (2019), CC BY 4.0.*
"""
)

code(
    r"""
# The same features, dtypes and split train.py uses, so the notebook and the script see identical rows
NUMERIC = [
    "lead_time", "stays_in_weekend_nights", "stays_in_week_nights", "adults", "children",
    "babies", "is_repeated_guest", "previous_cancellations", "previous_bookings_not_canceled",
    "days_in_waiting_list", "adr", "required_car_parking_spaces", "total_of_special_requests",
]
CATEGORICAL = [
    "hotel", "meal", "market_segment", "distribution_channel", "reserved_room_type",
    "deposit_type", "customer_type",
]
FEATURES = NUMERIC + CATEGORICAL
SEED = 42

X = bookings[FEATURES].astype({c: "float64" for c in NUMERIC})  # float64: see below
y = bookings["is_canceled"]
X_train, X_rest, y_train, y_rest = train_test_split(X, y, test_size=0.4, stratify=y, random_state=SEED)       # 60% train
X_val, X_test, y_val, y_test = train_test_split(X_rest, y_rest, test_size=0.5, stratify=y_rest, random_state=SEED)  # 20% / 20%

print(f"train {len(X_train):,} · validation {len(X_val):,} · test {len(X_test):,} · {len(FEATURES)} features")
"""
)

md(
    r"""
- **Train** fits, **validation** picks settings (P2), **test** scores each finished model once: `train.py` fits on train + validation and scores on test.
- **Numbers as `float64`:** the model's signature then says `double`, so a missing number reaches the imputer instead of being refused. The price is that callers must send floats; P4's service converts at the door.
"""
)

md(
    r"""
### `train.py` — the lines that matter

**One file, one tracked run; open `train.py` for the whole thing.**

```python
config = {"model_type": "xgboost", "n_estimators": args.n_estimators,
          "learning_rate": args.learning_rate, "max_depth": args.max_depth,
          "min_child_weight": args.min_child_weight}
if args.config:                           # P2 hands it a tuning run's best_config.json
    config.update(mlflow.artifacts.load_dict(args.config))
...
with mlflow.start_run(run_name=args.run_name) as run:
    mlflow.log_params(config)
    mlflow.log_dict(config, "config.json")
    mlflow.set_tags({"stage": "pipeline", "entry_point": "train.py"})
    ...                                   # the data file, logged as a dataset
    pipe = build_pipeline(config).fit(X_trainval, y_trainval)
    mlflow.log_metrics(evaluate(pipe, X_test, y_test, prefix="test"))
    model_info = mlflow.sklearn.log_model(
        pipe,
        name="model",
        signature=infer_signature(X_trainval, pipe.predict_proba(X_trainval)),
        input_example=X_trainval.head(5),
        pyfunc_predict_fn="predict_proba",  # generic callers get probabilities, not labels
        serialization_format="cloudpickle",
    )
```

- The defaults (`--n-estimators 300 --learning-rate 0.1 --max-depth 6 --min-child-weight 1`) are today's model: 300 trees, each at most 6 questions deep.
- The model is one scikit-learn `Pipeline`: preprocessing, then `XGBClassifier`. So `mlflow.sklearn` logs it, and every consumer gets the preprocessing with the model.
- `serialization_format="cloudpickle"`: MLflow's safer default, skops, refuses types this pipeline contains. The WARNING it prints is true: unpickling can run code, so load models only from a store you trust.
"""
)

code(
    r"""
# Train today's model with train.py's defaults: one tracked run in this notebook's store
!python train.py --run-name xgb-depth6
"""
)

md(
    r"""
- The run landed in this notebook's store without being told where: `set_tracking_uri` exported `MLFLOW_TRACKING_URI`, and `train.py` reads it.
- `python` here has to be the environment this kernel runs in (an activated venv or conda env).
"""
)

code(
    r"""
# The run train.py just logged, and the logged model it produced
current_run = mlflow.search_runs(
    experiment_names=["booking-cancellations"],
    filter_string="attributes.run_name = 'xgb-depth6'",
    order_by=["attributes.start_time DESC"],  # newest first, in case it was trained more than once
    max_results=1,
    output_format="list",                     # Run objects instead of a DataFrame
)[0]
current_model_id = current_run.outputs.model_outputs[0].model_id  # a run lists the models it logged

print("run      :", current_run.info.run_name, "|", current_run.info.run_id)
print("test     :", {k: round(v, 4) for k, v in current_run.data.metrics.items() if k.startswith("test_")})
print("size     :", round(current_run.data.metrics["model_size_mb"], 1), "MB")
print("model URI:", f"models:/{current_model_id}")
print("output   :", mlflow.models.get_model_info(f"models:/{current_model_id}").signature.outputs)
"""
)

md(
    r"""
### What that model doesn't have

**The model is tracked, but nothing says it's the one in use.**

- To use it, a consumer (the booking engine, a nightly scoring job) would be configured with `models:/m-…`: an ID.
- A better model next week means finding and changing every one of those configurations. Nothing records which model each consumer runs, who approved it, or what it replaced.
- P2 produces a better candidate. P3 gives both models a name, version numbers and roles.

💬 **Think about it** — Where does "which model is in production" live in your team today: a config file, a wiki page, a chat message? What happens when it's wrong?
"""
)

# ---------------------------------------------------------------- P2 : tune

part(
    "P2",
    "Tune: every trial a run",
    "search three settings with Optuna, one child run per trial",
    r"""
**One parent run for the search, one child run per trial: the search is a single row in the Runs table, and every trial in it can still be searched and compared.**

- The three settings: `learning_rate` (how much each new tree corrects the ones before it), `max_depth` (how many questions deep a tree may go), `min_child_weight` (how much evidence a split needs on each side: higher means more cautious trees).
- Optuna proposes values; each trial fits on train and scores on validation. The test set stays out of the search.
- **What this Part builds:** `build_pipeline()` and `evaluate()` (the same two functions `train.py` uses), `objective()`, the parent run `tune-xgb` with 20 child runs, and the challenger run `xgb-tuned`.

| New word | What it means here |
|---|---|
| study | one Optuna search: a direction (`maximize`) and its list of trials |
| trial | one proposed configuration and its score |
| objective | the function Optuna calls once per trial: configuration in, score out |
| TPE sampler | Optuna's default: 10 random trials first, then proposals near the best results so far |
| parent run / child run | runs linked by the tag `mlflow.parentRunId`; the UI folds the children under the parent |
| `nested=True` | `start_run` inside an open run creates a child of it, instead of refusing |
| `best_config.json` | the winning configuration as typed JSON, logged on the parent run |
| `runs:/<run_id>/<path>` | the address of one artifact of one run; `train.py --config` reads it |
""",
)

picture(
    "P2",
    "twenty runs, one answer",
    "The search is one parent run; its answer is a file the training script reads.",
    r"""
flowchart LR
    RAND["trial-00 … trial-09<br/>random proposals<br/>best ⟪p2_random_best⟫"]
    TPE["trial-10 … trial-19<br/>TPE proposals<br/>best ⟪p2_best_trial⟫: ⟪p2_best_val⟫"]
    CFG[("tune-xgb · best_config.json<br/>learning_rate ⟪p2_best_lr⟫<br/>max_depth ⟪p2_best_depth⟫ · min_child_weight ⟪p2_best_mcw⟫")]
    TRAIN["!python train.py --config<br/>runs:/…/best_config.json"]
    RUN[("run xgb-tuned<br/>test_roc_auc ⟪p2_tuned_test⟫ · ⟪p2_tuned_size⟫ MB")]
    RAND -- "then" --> TPE -- "best trial" --> CFG --> TRAIN --> RUN
""",
    r"""
    class RAND,TPE,TRAIN code
    class CFG,RUN stored
""",
    r"""
- 🟦 code and trials (each trial is a child run of `tune-xgb`) · 🟪 stored in MLflow.
- Validation ROC-AUC picks the configuration; test ROC-AUC is measured once, on the model `train.py` builds from it.
""",
)

code(
    r'''
# The two functions train.py uses, defined here too so every trial trains and scores exactly like the script
def build_pipeline(config):
    """Preprocessing + XGBoost, set up by `config`; returns an untrained Pipeline."""
    prep = ColumnTransformer([
        ("num", make_pipeline(SimpleImputer(strategy="median"), StandardScaler()), NUMERIC),  # fill gaps, rescale
        ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),                         # an unseen code → all zeros
    ])
    model = XGBClassifier(
        n_estimators=config["n_estimators"],          # how many trees, each one correcting the ones before it
        learning_rate=config["learning_rate"],
        max_depth=config["max_depth"],
        min_child_weight=config["min_child_weight"],
        random_state=SEED,
    )
    return Pipeline([("prep", prep), ("model", model)])


def evaluate(pipe, X, y, prefix):
    """Ranking quality (ROC-AUC, PR-AUC) plus F1 / precision / recall at the 0.5 threshold."""
    proba = pipe.predict_proba(X)[:, 1]
    pred = (proba >= 0.5).astype(int)
    return {
        f"{prefix}_roc_auc": roc_auc_score(y, proba),
        f"{prefix}_pr_auc": average_precision_score(y, proba),
        f"{prefix}_f1": f1_score(y, pred),
        f"{prefix}_precision": precision_score(y, pred),
        f"{prefix}_recall": recall_score(y, pred),
    }
'''
)

code(
    r'''
# One trial = one child run: propose a configuration, fit on train, score on validation, log it all
def objective(trial):
    """Optuna calls this once per trial; the value it returns is what the study maximises."""
    config = {
        "model_type": "xgboost",
        "n_estimators": 300,
        "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.3, log=True),  # log=True: sample evenly across ratios
        "max_depth": trial.suggest_int("max_depth", 3, 14),
        "min_child_weight": trial.suggest_float("min_child_weight", 1, 20, log=True),
    }
    with mlflow.start_run(run_name=f"trial-{trial.number:02d}", nested=True):  # a child of the open parent run
        mlflow.log_params(config)
        mlflow.set_tag("stage", "trial")
        start = time.time()
        pipe = build_pipeline(config).fit(X_train, y_train)
        fit_time = time.time() - start
        metrics = evaluate(pipe, X_val, y_val, prefix="val")
        mlflow.log_metrics({**metrics, "fit_time_s": fit_time, "model_size_mb": len(pickle.dumps(pipe)) / 1e6})

    print(f"trial-{trial.number:02d}  val_roc_auc {metrics['val_roc_auc']:.4f}  {fit_time:4.1f}s  "
          f"learning_rate {config['learning_rate']:.3f}  max_depth {config['max_depth']:2d}  "
          f"min_child_weight {config['min_child_weight']:5.2f}")
    return metrics["val_roc_auc"]
'''
)

md(
    r"""
### Run the search

**The parent run opens first; every `start_run(nested=True)` inside `objective()` becomes its child.**

- 20 trials, about ⟪p2_seconds⟫ seconds. Watch the UI while it runs (next section): the children appear as they finish.
- The parent records the question (sampler, number of trials) and the answer (`best_val_roc_auc`, `best_config.json`, a plot of the search).
"""
)

code(
    r"""
# The search: one parent run; study.optimize() calls objective() 20 times, each call a child run inside it
optuna.logging.set_verbosity(optuna.logging.WARNING)  # objective() prints its own one-line summary per trial

with mlflow.start_run(
    run_name="tune-xgb",
    description="Optuna TPE over learning_rate, max_depth, min_child_weight; fit on train, scored on validation.",
) as tuning_run:
    mlflow.set_tag("stage", "tuning")
    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=SEED))
    study.optimize(objective, n_trials=20)

    best_config = {"model_type": "xgboost", "n_estimators": 300, **study.best_params}  # typed values, ready for train.py
    mlflow.log_params({"n_trials": len(study.trials), "sampler": "TPE", "seed": SEED})
    mlflow.log_metric("best_val_roc_auc", study.best_value)
    mlflow.log_dict(best_config, "best_config.json")

    history = study.trials_dataframe()  # one row per trial: its number, value and params
    fig, ax = plt.subplots(figsize=(8, 3.2))
    ax.plot(history.number, history.value, "o", label="trial")
    ax.plot(history.number, history.value.cummax(), "-", label="best so far")
    ax.axvline(9.5, color="grey", linestyle=":", label="random → TPE")
    ax.set(xlabel="trial", ylabel="val_roc_auc", title="tune-xgb", xticks=range(0, 20, 2))
    ax.legend()
    mlflow.log_figure(fig, "plots/search_history.png")

tuning_run_id = tuning_run.info.run_id
print(f"\nbest: trial-{study.best_trial.number:02d} · val_roc_auc {study.best_value:.4f} · {best_config}")
"""
)

md(
    r"""
### In the UI

**Runs → `tune-xgb` is one row, with a `+` beside its name.**

- Click the `+` to unfold the 20 trials under it; sort by `val_roc_auc`.
- Tick the trials → **Compare** → **Parallel Coordinates Plot** with `learning_rate`, `max_depth`, `min_child_weight` → `val_roc_auc`: which settings do the best trials share?
- `tune-xgb` → **Artifacts**: `best_config.json` and `plots/search_history.png`.
"""
)

code(
    r"""
# The trials as a leaderboard: children are found through the tag MLflow sets on them, mlflow.parentRunId
trials = mlflow.search_runs(
    experiment_names=["booking-cancellations"],
    filter_string=f"tags.mlflow.parentRunId = '{tuning_run_id}'",
    order_by=["metrics.val_roc_auc DESC"],
)

top = trials.head(6)
pd.DataFrame({
    "run": top["tags.mlflow.runName"],
    "learning_rate": top["params.learning_rate"].astype(float).round(3),  # params come back as strings
    "max_depth": top["params.max_depth"],
    "min_child_weight": top["params.min_child_weight"].astype(float).round(2),
    "val_roc_auc": top["metrics.val_roc_auc"].round(4),
    "fit_time_s": top["metrics.fit_time_s"].round(1),
    "model_size_mb": top["metrics.model_size_mb"].round(1),
})
"""
)

md(
    r"""
**⟪p2_pattern_title⟫**

- ⟪p2_pattern_1⟫
- ⟪p2_pattern_2⟫
- Validation ROC-AUC rose from ⟪p2_random_best⟫ (best random trial) to ⟪p2_best_val⟫. The best of 20 validation scores is a little optimistic by construction, which is why the test set decides in P5.
"""
)

md(
    r"""
### The hand-off: tuning run → training job

**`train.py` reads the winning configuration straight from the tuning run, so nobody retypes numbers.**

- `--config runs:/<tuning_run_id>/best_config.json` → `mlflow.artifacts.load_dict(...)` inside `train.py`. The values stay typed: ⟪p2_best_depth⟫ is still an int.
- `train.py` records where its config came from, as the tag `config_source`.
- `$tuning_run_id` in a `!` line is the Python variable: IPython fills it in before the shell runs.
"""
)

code(
    r"""
# Train the challenger: same script, same data and split, the configuration the search picked
!python train.py --config runs:/$tuning_run_id/best_config.json --run-name xgb-tuned
"""
)

code(
    r"""
# Both train.py runs side by side: only the configuration differs
mlflow.search_runs(
    experiment_names=["booking-cancellations"],
    filter_string="tags.stage = 'pipeline'",
    order_by=["attributes.start_time ASC"],
)[["tags.mlflow.runName", "params.max_depth", "params.min_child_weight", "params.learning_rate",
   "metrics.test_roc_auc", "metrics.model_size_mb", "tags.config_source"]]
"""
)

md(
    r"""
- `xgb-tuned`: test ROC-AUC ⟪p2_tuned_test⟫ against ⟪p1_test⟫, at ⟪p2_tuned_size⟫ MB instead of ⟪p1_size⟫. Whether that's worth replacing today's model is P5's question.
- Both are logged models with IDs. Neither is "the" model yet.

💬 **Think about it** — The search tuned three settings against one validation split and kept the best of 20 scores. What could make its winner look better than it really is, and what in this notebook guards against that?

✍️ **Try it** — ⟪p2_try_question⟫

<details><summary>Answer</summary>

```python
mlflow.search_runs(
    experiment_names=["booking-cancellations"],
    filter_string=f"tags.mlflow.parentRunId = '{tuning_run_id}' "
                  "⟪p2_try_filter⟫",
)[["tags.mlflow.runName", "params.max_depth", "params.learning_rate",
   "metrics.val_roc_auc", "metrics.model_size_mb"]]
```

⟪p2_try_answer⟫

</details>
"""
)

# ---------------------------------------------------------------- P3 : register

part(
    "P3",
    "Register: a name, versions, aliases",
    "give both models one name, version numbers and roles",
    r"""
**The Model Registry turns "the model from run `3df2…`" into `booking-cancellation` version 2, and "the one in production" into an alias anyone can look up, or move.**

- A registered model is a name. Each registration adds the next version number, which never changes and is never reused.
- The name says what the model is for, not which algorithm it is. Both versions here are XGBoost; one built with another library would still be `booking-cancellation`, loaded from the same URI.
- An alias points at one version. Consumers load `models:/booking-cancellation@champion` and never name a version themselves.
- **What this Part builds:** the registered model `booking-cancellation`: version 1 (`xgb-depth6`) as `@champion`, version 2 (`xgb-tuned`) as `@challenger`, with descriptions and a tag.

| New word | What it means here |
|---|---|
| Model Registry | MLflow's catalogue of named models, in the same store as the runs |
| registered model | a name that holds numbered versions: `booking-cancellation` |
| model version | one logged model registered under that name: 1, 2, … |
| alias | a movable label naming exactly one version at a time: `@champion`, `@challenger` |
| `models:/<name>/<version>` | a fixed version: always the same model |
| `models:/<name>@<alias>` | whichever version the alias points at when the model is loaded |
| version description / tag | notes on a version: what it is, where it came from |
| stages | the older `Staging` / `Production` labels; deprecated since MLflow 2.9, replaced by aliases |
""",
    # a framing question added by hand in the plain folder's v1 notebook (2026-09-30), kept word for word
    question=(None if AIRFLOW else
              "“Out of all these trained models, which model is actually the official model, "
              "which one is the challenger, and how should applications refer to them?”"),
)

picture(
    "P3",
    "two versions, two roles",
    "Registering gives each logged model a version number; aliases give versions roles.",
    r"""
flowchart LR
    R1[("run xgb-depth6<br/>models:/m-…")]
    R2[("run xgb-tuned<br/>models:/m-…")]
    V1[("booking-cancellation<br/>version 1")]
    V2[("booking-cancellation<br/>version 2")]
    CH(["@champion"])
    CL(["@challenger"])
    USE(["consumers load<br/>models:/booking-cancellation@champion"])
    R1 -- "register_model" --> V1 --- CH
    R2 -- "register_model" --> V2 --- CL
    CH --> USE
""",
    r"""
    class R1,R2,V1,V2 stored
    class CH,CL person
    class USE endpoint
""",
    r"""
- 🟪 stored in MLflow · 🟧 aliases: a role someone decides · 🟩 the consumers.
- Consumers hold the alias, never the version: moving `@champion` (P5) changes what they load.
""",
)

code(
    r"""
# Register both logged models under one name: each call creates the next version number
REGISTERED_MODEL = "booking-cancellation"  # what the model is for, not which algorithm it is

tuned_run = mlflow.search_runs(
    experiment_names=["booking-cancellations"],
    filter_string="attributes.run_name = 'xgb-tuned'",
    order_by=["attributes.start_time DESC"],
    max_results=1,
    output_format="list",
)[0]
tuned_model_id = tuned_run.outputs.model_outputs[0].model_id

v1 = mlflow.register_model(f"models:/{current_model_id}", REGISTERED_MODEL)  # xgb-depth6
v2 = mlflow.register_model(f"models:/{tuned_model_id}", REGISTERED_MODEL)    # xgb-tuned

for mv in (v1, v2):
    print(f"version {mv.version}: source {mv.source} · run {mv.run_id} · status {mv.status}")
"""
)

md(
    r"""
- **`source`** is the logged model this version is; **`run_id`** is the run that trained it. From a version, the params, metrics and data are one hop away.
- Register the logged model's own URI, `models:/m-…`. A `runs:/<run_id>/model` URI works too, with a warning that MLflow swapped in the logged model.
- In a training script, `log_model(..., registered_model_name="booking-cancellation")` registers at logging time, so every run becomes a version. Registering by hand, as here, keeps the registry for candidates.
"""
)

code(
    r"""
# Roles and notes: @champion is what consumers load today, @challenger what wants to replace it
client.set_registered_model_alias(REGISTERED_MODEL, "champion", v1.version)
client.set_registered_model_alias(REGISTERED_MODEL, "challenger", v2.version)

client.update_registered_model(
    REGISTERED_MODEL,
    description="P(cancel) for one hotel booking. Loaded as @champion by the booking engine's risk service.",
)
client.update_model_version(REGISTERED_MODEL, v1.version, description="XGBoost, 300 trees of depth 6: the configuration running today.")
client.update_model_version(REGISTERED_MODEL, v2.version, description="Tuned by Optuna: 20 trials on validation ROC-AUC.")
client.set_model_version_tag(REGISTERED_MODEL, v2.version, "tuning_run", tuning_run_id)  # the search that produced it

client.get_registered_model(REGISTERED_MODEL).aliases  # {alias: version}
"""
)

code(
    r"""
# The registry's view: every version, its role, and (one hop away) the run and the numbers behind it
aliases = client.get_registered_model(REGISTERED_MODEL).aliases  # search results below leave `aliases` empty; this map doesn't

rows = []
for mv in client.search_model_versions(f"name = '{REGISTERED_MODEL}'"):
    run = mlflow.get_run(mv.run_id)
    rows.append({
        "version": mv.version,
        "aliases": [alias for alias, version in aliases.items() if int(version) == int(mv.version)],
        "run": run.info.run_name,
        "test_roc_auc": round(run.data.metrics["test_roc_auc"], 4),
        "model_size_mb": round(run.data.metrics["model_size_mb"], 1),
        "description": mv.description,
    })
pd.DataFrame(rows).sort_values("version")
"""
)

md(
    r"""
### In the UI

**Sidebar → Model registry → `booking-cancellation`.**

- The **Versions** table: **Version · Registered at · Created by · Tags · Aliases · Description**. The pencil in the **Aliases** cell edits a version's aliases.
- Click **Version 2**: **Source Run** links back to `xgb-tuned`, and **Schema** lists its 20 inputs and 1 output.
- **Stage (deprecated): None** is the old mechanism, left in the page.
- **Runs** → the **Models** column now reads `booking-cancellation v1` / `v2` beside the two `train.py` runs.
"""
)

code(
    r"""
# A consumer names the role, never a run or a version: this line stays the same when the champion changes
champion_model = mlflow.pyfunc.load_model(f"models:/{REGISTERED_MODEL}@champion")

print("@champion is version", client.get_model_version_by_alias(REGISTERED_MODEL, "champion").version,
      "· logged model", champion_model.metadata.model_id)
pd.DataFrame(champion_model.predict(X_test.head(3)), columns=["P(kept)", "P(cancelled)"]).round(4)  # one row per booking
"""
)

md(
    r"""
- Two columns per row because `train.py` logged the model with `pyfunc_predict_fn="predict_proba"`.
- `get_model_version_by_alias` answers "which version is `@champion` right now?" without loading anything; P4's service uses it.

💬 **Think about it** — Who should be allowed to move `@champion`: the training job itself, a reviewer, or a CI job after automated checks? What would you want recorded each time it moves?

✍️ **Try it** — Load version 2 once by number and once by alias, and confirm they predict the same.

<details><summary>Answer</summary>

```python
by_number = mlflow.pyfunc.load_model(f"models:/{REGISTERED_MODEL}/2")
by_alias = mlflow.pyfunc.load_model(f"models:/{REGISTERED_MODEL}@challenger")
(by_number.predict(X_test) == by_alias.predict(X_test)).all()
```

</details>
"""
)

# ---------------------------------------------------------------- P4 : serve

part(
    "P4",
    "Serve: one command, then a contract",
    "put @champion behind HTTP twice: MLflow's own server, then a service that checks values",
    r"""
**`mlflow models serve` puts any registered model behind a REST endpoint with no code. Its only check is the model's signature: right names, right types. Right values need a contract of their own.**

- **What this Part builds:** MLflow's scoring server around `@champion`; then `app.py`'s three pieces: `Booking` (the value contract), `load_champion()`, and the routes `/predict` and `/health`.
- Every request here goes through `TestClient`: the real routing and validation, in this process, no port. Each step has its terminal command beside it, for the real server.

| New word | What it means here |
|---|---|
| `mlflow models serve` | a CLI command that starts a web server for one model URI |
| scoring server | the FastAPI app that command runs under uvicorn; `scoring_server.init(model)` builds the same app in this process |
| `/invocations` | its POST route: rows in, `{"predictions": …}` out |
| `dataframe_records` | one of its input formats: a JSON list of `{column: value}` rows |
| `--env-manager local` | serve from the current Python environment instead of building one from the model's `python_env.yaml` |
| `TestClient` | sends requests to an app inside this process: the same routing, validation and errors as over HTTP |
| route | a URL + method handled by one function: `@app.post("/predict")` |
| value contract | the values each field may take, checked before the model runs |
| `Literal[...]`, `Field(ge=0)` | pydantic: exactly one of these values · a number ≥ 0 |
| 400 / 422 | MLflow's "doesn't match the signature" / FastAPI's "doesn't match the schema" |
""",
)

picture(
    "P4",
    "six requests, two doors",
    "The signature stops wrong names and types; the value contract also stops wrong values.",
    r"""
flowchart LR
    REQ(["BK218304<br/>as sent, and five mistakes"])
    SIG{{"MLflow /invocations<br/>signature: names and types"}}
    S200(["200 scored<br/>as sent ⟪p4_as_sent⟫<br/>'Online Travel Agent' ⟪p4_ota⟫<br/>lead_time -1: ⟪p4_minus1⟫<br/>'city hotel' ⟪p4_city⟫"])
    S400(["400<br/>'five months' in lead_time<br/>deposit_type missing"])
    VAL{{"app.py /predict<br/>Booking: codes and ranges"}}
    A422(["422, the field named<br/>all five mistakes"])
    A200(["200 · as sent<br/>p_cancel ⟪p4_as_sent⟫ · reconfirm · version 1"])
    REQ --> SIG
    SIG --> S200
    SIG --> S400
    REQ --> VAL
    VAL --> A422
    VAL --> A200
""",
    r"""
    class REQ,S200,S400,A422,A200 endpoint
    class SIG,VAL guard
""",
    r"""
- 🟩 requests and answers · 🟥 the two checks.
- Three of MLflow's 200s are wrong answers delivered with full confidence; ⟪p4_flip_count⟫ of them flip a reconfirm call.
""",
)

md(
    r"""
### `mlflow models serve` — in a terminal

**One command serves the champion. It resolves `@champion` once, when it starts.**

```bash
cd "<the folder that contains this notebook>"
MLFLOW_TRACKING_URI=sqlite:///mlflow.db mlflow models serve \
  -m "models:/booking-cancellation@champion" --env-manager local --port 5002
```

- `MLFLOW_TRACKING_URI` tells it which store holds the registry. `--port 5002`: 5001 is the UI.
- Its log names what it actually runs: uvicorn serving `mlflow.pyfunc.scoring_server.app:app`, the app the next cell builds in this process.
"""
)

code(
    r"""
# The app `mlflow models serve` runs, built here around the champion: the requests below need no running server
logging.getLogger("mlflow.pyfunc.scoring_server").setLevel(logging.ERROR)  # on every 400 it logs a hint about a format not used here

model_server = TestClient(scoring_server.init(champion_model))
print("GET /ping →", model_server.get("/ping").status_code)
"""
)

code(
    r"""
# BK218304, from next week's arrivals: the 20 columns the signature names, as plain JSON values
upcoming = pd.read_csv("data/upcoming_arrivals.csv")
sample_booking = json.loads(upcoming.loc[upcoming.booking_id == "BK218304", FEATURES].to_json(orient="records"))[0]

response = model_server.post("/invocations", json={"dataframe_records": [sample_booking]})
print(response.status_code, response.json())
sample_booking
"""
)

md(
    r"""
- **200** and one pair per row, `[P(kept), P(cancelled)]`. BK218304 is at ⟪p4_as_sent_2dp⟫: the revenue team would reconfirm it.
- The server builds its DataFrame from the JSON using the signature's types, so `"lead_time": 159` arrives as a float. An int64 DataFrame passed to `predict()` in Python would be refused.
- The same request from a terminal, while the server above runs:

```bash
curl -s -X POST http://127.0.0.1:5002/invocations -H 'Content-Type: application/json' \
  -d '{"dataframe_records": [{
    "lead_time": 159, "stays_in_weekend_nights": 0, "stays_in_week_nights": 2,
    "adults": 3, "children": 0, "babies": 0, "is_repeated_guest": 0,
    "previous_cancellations": 0, "previous_bookings_not_canceled": 0,
    "days_in_waiting_list": 0, "adr": 175.5, "required_car_parking_spaces": 0,
    "total_of_special_requests": 0, "hotel": "City Hotel", "meal": "BB",
    "market_segment": "Online TA", "distribution_channel": "TA/TO",
    "reserved_room_type": "D", "deposit_type": "No Deposit", "customer_type": "Transient"}]}'
```
"""
)

md(
    r"""
### What the signature lets through

**Five requests a real caller could send, each with one mistake in it.**

- The name of a code instead of the code; an upstream system's `-1` for "unknown"; a value lower-cased on the way; text in a number field; a column dropped.
"""
)

code(
    r"""
# Six requests to MLflow's server: BK218304 as sent, then five mistakes a real caller could make
variants = {
    "as sent": sample_booking,
    "market_segment = 'Online Travel Agent'": {**sample_booking, "market_segment": "Online Travel Agent"},  # the name, not the code
    "lead_time = -1": {**sample_booking, "lead_time": -1},                                                 # an upstream "unknown"
    "hotel = 'city hotel'": {**sample_booking, "hotel": "city hotel"},                                     # lower-cased on the way
    "lead_time = 'five months'": {**sample_booking, "lead_time": "five months"},                           # text in a number field
    "deposit_type missing": {k: v for k, v in sample_booking.items() if k != "deposit_type"},             # a column dropped upstream
}

for label, record in variants.items():
    response = model_server.post("/invocations", json={"dataframe_records": [record]})
    if response.status_code == 200:
        print(f"{response.status_code}  {label:40} p_cancel {response.json()['predictions'][0][1]:.4f}")
    else:  # MLflow's message ends with the reason, after "Error: "
        print(f"{response.status_code}  {label:40} {response.json()['message'].rsplit('Error: ', 1)[-1][:70]}")
"""
)

md(
    r"""
**The signature stopped the two mistakes it can see, names and types, and scored the three it can't.**

- **400** · text in `lead_time`, a missing `deposit_type`: refused before the model runs.
- ⟪p4_ota_line⟫
- ⟪p4_minus1_line⟫
- ⟪p4_city_line⟫
- A signature is a contract about names and types. Values need their own contract, in front of the model.
"""
)

md(
    r"""
### The service: `app.py`

**`app.py` is what the booking engine calls: its own request shape, a value contract, and the model loaded through the alias.**

- `Booking` lists every field the engine sends, with the codes from P1's table as `Literal[...]` and counts as `Field(ge=0)`. FastAPI checks each request against it before the route runs. A failed check is a 422 that names the field.
- `load_champion()` resolves `@champion` to a version number once, then loads exactly that version. The answer and the version it reports can't disagree, even if the alias moves while the service runs.
- `/predict` returns the decision and its basis: `p_cancel`, `reconfirm`, the `threshold` it applied, and `model_version`.
"""
)

BOOKING_SRC = r'''
# app.py, part 1: the value contract. FastAPI validates every request body against it before the route runs
class Booking(BaseModel):
    """One booking, as the booking engine sends it: codes the model was trained on, counts ≥ 0."""

    booking_id: str
    hotel: Literal["City Hotel", "Resort Hotel"]
    lead_time: int = Field(ge=0)
    stays_in_weekend_nights: int = Field(ge=0)
    stays_in_week_nights: int = Field(ge=0)
    adults: int = Field(ge=0)
    children: int = Field(ge=0)
    babies: int = Field(ge=0)
    meal: Literal["BB", "HB", "FB", "SC", "Undefined"]
    market_segment: Literal[
        "Online TA", "Offline TA/TO", "Groups", "Direct", "Corporate", "Complementary", "Aviation", "Undefined"
    ]
    distribution_channel: Literal["TA/TO", "Direct", "Corporate", "GDS", "Undefined"]
    is_repeated_guest: Literal[0, 1]
    previous_cancellations: int = Field(ge=0)
    previous_bookings_not_canceled: int = Field(ge=0)
    reserved_room_type: Literal["A", "B", "C", "D", "E", "F", "G", "H", "L", "P"]
    deposit_type: Literal["No Deposit", "Non Refund", "Refundable"]
    customer_type: Literal["Transient", "Transient-Party", "Contract", "Group"]
    days_in_waiting_list: int = Field(ge=0)
    adr: float = Field(ge=0)
    required_car_parking_spaces: int = Field(ge=0)
    total_of_special_requests: int = Field(ge=0)
'''

LOAD_SRC = r'''
# app.py, part 2: which model answers. Resolve @champion to a version once, then load exactly that version
THRESHOLD = 0.5  # the revenue team reconfirms a booking at P(cancel) ≥ 0.5


def load_champion():
    """Resolve `@champion` to a version number once, then load exactly that version."""
    version = MlflowClient().get_model_version_by_alias(REGISTERED_MODEL, "champion").version
    model = mlflow.pyfunc.load_model(f"models:/{REGISTERED_MODEL}/{version}")
    return model, version


model, model_version = load_champion()  # once, at startup
print("answering with", REGISTERED_MODEL, "version", model_version)
'''

ROUTES_SRC = r'''
# app.py, part 3: the routes. FastAPI builds `booking` from the JSON body (or answers 422) before predict() runs
app = FastAPI(title="Maré Hotels · cancellation risk")


@app.get("/health")
def health():
    """Up, and answering with this model version."""
    return {"status": "ok", "model": REGISTERED_MODEL, "model_version": model_version}


@app.post("/predict")
def predict(booking: Booking):
    """Score one booking. FastAPI has already validated the body against `Booking` (or answered 422)."""
    # one-row DataFrame in the model's column order; numbers as float64, which the signature expects
    frame = pd.DataFrame([booking.model_dump()])[FEATURES].astype({c: "float64" for c in NUMERIC})
    p_cancel = float(model.predict(frame)[0, 1])  # predict returns [P(kept), P(cancelled)] per row
    return {
        "booking_id": booking.booking_id,
        "p_cancel": round(p_cancel, 4),
        "reconfirm": p_cancel >= THRESHOLD,
        "threshold": THRESHOLD,
        "model_version": model_version,
    }


service = TestClient(app)  # requests to app.py's routes, in this process
print("GET /health →", service.get("/health").json())
service.post("/predict", json={"booking_id": "BK218304", **sample_booking}).json()
'''

code(BOOKING_SRC)
code(LOAD_SRC)
code(ROUTES_SRC)

md(
    r"""
- Same model, same booking, same ⟪p4_as_sent⟫ as MLflow's server, now with the decision (`reconfirm`), the rule behind it (`threshold`) and the version that made it.
- `model_dump()` turns the validated `Booking` into a plain dict; the `astype` line casts the numbers to float64, as the signature asks.
"""
)

code(
    r'''
# The same six requests to both endpoints: ✅ the mistake was stopped · ❌ it was scored anyway
def outcome(label, response, score):
    """One table cell: status code, then the score (a 200) or why it was refused."""
    if response.status_code == 200:
        mark = "" if label == "as sent" else "❌ "  # a 200 is right only for the booking as sent
        return f"{mark}200 · p_cancel {score(response.json()):.4f}"
    return f"✅ {response.status_code}"


rows = []
for label, record in variants.items():
    mlflow_response = model_server.post("/invocations", json={"dataframe_records": [record]})
    app_response = service.post("/predict", json={"booking_id": "BK218304", **record})
    rows.append({
        "request": label,
        "MLflow /invocations": outcome(label, mlflow_response, lambda body: body["predictions"][0][1]),
        "app.py /predict": outcome(label, app_response, lambda body: body["p_cancel"]),
        "field the 422 names": ", ".join(error["loc"][-1] for error in app_response.json().get("detail", [])),
    })
pd.DataFrame(rows)
'''
)

md(
    r"""
**`app.py` stopped all five mistakes at the door, and each 422 names the field, so the caller knows what to fix.**

- MLflow's server is the right tool when the caller already speaks in model columns: another team's batch job, a quick check of a new version, a platform that serves many models the same way.
- `app.py` is the right tool when the caller is a product: it adds the allowed values, the decision rule and the version to report.
- No schema catches a plausible wrong value: `lead_time` sent in weeks (23 instead of 159) passes both doors.

💬 **Think about it** — A caller starts sending `lead_time` in weeks. Both endpoints accept it. Where, and how soon, would you notice?
"""
)

md(
    r"""
### Run it for real — in a terminal

**`app.py` holds exactly the code of the three cells above, plus the constants they use.**

```bash
cd "<the folder that contains this notebook>"
uvicorn app:app --port 8001
```

- Open http://127.0.0.1:8001/docs → **POST /predict** → **Try it out**. The form is generated from `Booking`, allowed values included.
- `curl -s http://127.0.0.1:8001/health` reports the version answering.

| In this notebook | In `app.py` |
|---|---|
| `NUMERIC`, `CATEGORICAL`, `FEATURES` (P1), `REGISTERED_MODEL` (P3) | the same constants, at the top |
| `mlflow.set_tracking_uri("sqlite:///mlflow.db")` (P0) | `MLFLOW_TRACKING_URI` from the environment, `sqlite:///mlflow.db` by default |
| `Booking`, `THRESHOLD`, `load_champion()`, `app`, `health()`, `predict()` (P4) | identical |
| `model_server`, `service` (the `TestClient`s) | not there: uvicorn serves the app instead |
"""
)

# ---------------------------------------------------------------- P5 : promote

part(
    "P5",
    "Promote: move the alias",
    "decide with a gate, move @champion, restart, and keep the way back",
    r"""
**Promotion is one call. The work is deciding when to make it, and being able to undo it.**

- The gate compares both versions on the same test set, through the runs the registry points at.
- **What this Part builds:** the gate in code, a check of what promotion changes for next week's arrivals, the alias move, a restart of both endpoints, and the rollback as a ready call.

| New word | What it means here |
|---|---|
| promotion gate | the rule a challenger must pass before it becomes champion: here +0.002 test ROC-AUC or more, under 50 MB |
| promote | point `@champion` at the challenger's version |
| restart | both endpoints resolve `@champion` when they start, so the next start picks up the new version |
| rollback | point `@champion` back at the previous version, then restart |
| decision flip | a booking whose reconfirm yes/no differs between the two versions |
""",
)

picture(
    "P5",
    "one alias move",
    "The gate reads both versions' numbers; passing it moves one alias, and a restart does the rest.",
    r"""
flowchart LR
    V1[("version 1 @champion<br/>test_roc_auc ⟪p1_test⟫")]
    V2[("version 2 @challenger<br/>test_roc_auc ⟪p2_tuned_test⟫ · ⟪p2_tuned_size⟫ MB")]
    GATE{{"gate<br/>gain ⟪p5_gain⟫, at least 0.002<br/>⟪p2_tuned_size⟫ MB, under 50 MB"}}
    MOVE["set_registered_model_alias<br/>champion → version 2"]
    RESTART(["restart both endpoints<br/>/health: model_version 2<br/>BK218304: ⟪p5_after⟫"])
    BACK(["rollback<br/>champion → version 1, restart"])
    V1 --> GATE
    V2 --> GATE
    GATE -- "passed" --> MOVE --> RESTART
    RESTART -. "if version 2 misbehaves" .-> BACK
""",
    r"""
    class V1,V2 stored
    class GATE guard
    class MOVE code
    class RESTART endpoint
    class BACK person
""",
    r"""
- 🟪 stored · 🟥 the gate · 🟦 the one call · 🟩 what callers see · 🟧 the way back.
- Nothing on the serving side changes: not `app.py`, not a command line, not a config file.
""",
)

code(
    r"""
# The promotion gate: the challenger must beat the champion on the same test set, within the size budget
MIN_GAIN = 0.002     # a smaller gain is too close to test-set noise to be worth a change
SIZE_BUDGET_MB = 50  # the platform team's cap on model artifacts

champion = client.get_model_version_by_alias(REGISTERED_MODEL, "champion")
challenger = client.get_model_version_by_alias(REGISTERED_MODEL, "challenger")
champion_metrics = mlflow.get_run(champion.run_id).data.metrics      # version → run → what train.py logged
challenger_metrics = mlflow.get_run(challenger.run_id).data.metrics

gain = challenger_metrics["test_roc_auc"] - champion_metrics["test_roc_auc"]
gate = {
    f"test_roc_auc {challenger_metrics['test_roc_auc']:.4f} vs {champion_metrics['test_roc_auc']:.4f}: gain {gain:+.4f} ≥ {MIN_GAIN}": gain >= MIN_GAIN,
    f"model_size_mb {challenger_metrics['model_size_mb']:.1f} < {SIZE_BUDGET_MB}": challenger_metrics["model_size_mb"] < SIZE_BUDGET_MB,
}
for check, passed in gate.items():
    print("✅" if passed else "❌", check)
"""
)

code(
    r"""
# What promotion would change for the business: next week's arrivals, scored by both versions
challenger_model = mlflow.pyfunc.load_model(f"models:/{REGISTERED_MODEL}@challenger")
arrivals = upcoming[FEATURES].astype({c: "float64" for c in NUMERIC})
p_champion = champion_model.predict(arrivals)[:, 1]
p_challenger = challenger_model.predict(arrivals)[:, 1]
flips = (p_champion >= THRESHOLD) != (p_challenger >= THRESHOLD)

print(f"reconfirm list: {(p_champion >= THRESHOLD).sum()} bookings (version {champion.version}) → {(p_challenger >= THRESHOLD).sum()} (version {challenger.version})")
print(f"decision flips: {flips.sum()} of {len(arrivals):,} · off the list {((p_champion >= THRESHOLD) & flips).sum()} · onto it {((p_challenger >= THRESHOLD) & flips).sum()}")

upcoming.loc[flips, ["booking_id", "arrival_date", "hotel", "lead_time", "market_segment"]].assign(
    p_champion=p_champion[flips].round(3), p_challenger=p_challenger[flips].round(3)
).head(5)
"""
)

md(
    r"""
- ⟪p5_flips_line⟫
- That's what the revenue team feels on promotion day. A gate says the new model is better; this says what's different, and who should hear about it.
"""
)

code(
    r"""
# Promote only if the gate passed: move @champion, retire @challenger, and record the decision on the version
if all(gate.values()):
    previous_champion = champion.version  # the rollback target
    client.set_registered_model_alias(REGISTERED_MODEL, "champion", challenger.version)
    client.delete_registered_model_alias(REGISTERED_MODEL, "challenger")
    client.set_model_version_tag(
        REGISTERED_MODEL, challenger.version, "promoted_over", f"version {champion.version}, test_roc_auc {gain:+.4f}"
    )
else:
    print("gate failed: @champion stays on version", champion.version)

client.get_registered_model(REGISTERED_MODEL).aliases
"""
)

code(
    r"""
# "Restart" both endpoints: each resolves @champion again when it starts. No code or config changed
model, model_version = load_champion()  # what app.py's startup line does
model_server = TestClient(scoring_server.init(mlflow.pyfunc.load_model(f"models:/{REGISTERED_MODEL}@champion")))  # what `mlflow models serve` does

print("app.py /health      :", service.get("/health").json())
print("app.py /predict     :", service.post("/predict", json={"booking_id": "BK218304", **sample_booking}).json())
print("MLflow /invocations :", model_server.post("/invocations", json={"dataframe_records": [sample_booking]}).json())
"""
)

md(
    r"""
- In a terminal the restart is Ctrl+C and the same command again, for `uvicorn` and for `mlflow models serve`.
- Both endpoints now answer from version 2: BK218304 moves from ⟪p4_as_sent⟫ to ⟪p5_after⟫, still a reconfirm.
- MLflow shows where an alias points now, not where it pointed before. The `promoted_over` tag on version 2 is the record of this move.

### Who moves the alias

**In the UI, and in a team.**

- **UI:** Model registry → `booking-cancellation` → the pencil in a version's **Aliases** cell → **Add/Edit alias** → **Save aliases**. An alias lives on one version at a time: typing `champion` on another version shows *"The "champion" alias is also being used on version 1. Adding it to this version will remove it from version 1."*
- The version page's **Promote model** button is something else: it copies the version into another registered model (one registered model per environment).
- **In a team:** """
    + ("a scheduled job runs `train.py` and the gate (P6); a reviewer, or the job itself, moves the alias."
       if AIRFLOW else
       "a CI job runs `train.py` and the gate; a reviewer, or the job itself, moves the alias.")
    + r""" Only the gate in code makes the decision repeatable.

✍️ **Try it** — Version 2 misbehaves in production. Roll back, restart, and check `/health` reports version 1; then promote version 2 again.

<details><summary>Answer</summary>

```python
client.set_registered_model_alias(REGISTERED_MODEL, "champion", previous_champion)  # roll back
model, model_version = load_champion()                                              # restart
print(service.get("/health").json())                                                # version 1

# forward again
client.set_registered_model_alias(REGISTERED_MODEL, "champion", challenger.version)
model, model_version = load_champion()
```

</details>
"""
)

# ---------------------------------------------------------------- P6 : package (plain folder only)

if not AIRFLOW:
    # the two project files are quoted in full, straight from disk, so the notebook can't drift from them
    MLPROJECT_SRC = (HERE / "MLproject").read_text().strip("\n")
    PYTHON_ENV_SRC = (HERE / "python_env.yaml").read_text().strip("\n")

    part(
        "P6",
        "Package: a project anyone can run",
        "give the training job named commands and a declared environment, then rebuild the champion with one",
        r"""
**`train.py` runs because whoever starts it knows its flags, its environment and where the runs go. An MLflow Project writes those down in the folder, so a teammate, or a scheduler, starts the job with one command.**

- An MLflow Project is this folder plus one file, `MLproject`: the commands the project offers and the parameters each takes. A second file, `python_env.yaml`, names the environment they need.
- `mlflow run` reads both, opens a run, and starts the command inside it.
- **What this Part builds:** `MLproject` with two entry points, `train` and `retrain`; `python_env.yaml`; the runs `project-train` and `project-retrain`; and with the second, the champion rebuilt from its own `config.json`.

| New word | What it means here |
|---|---|
| MLflow Project | a folder, or a Git repository, with an `MLproject` file at its root |
| `MLproject` | a YAML file: the project's name, its environment and its entry points |
| entry point | a command the project offers, by name: `train`, `retrain` |
| parameter | a named value an entry point takes, with a type label and an optional default |
| `mlflow run` | the CLI command that runs one entry point inside a tracked run |
| `-e` · `-P name=value` | which entry point · a value for one of its parameters |
| `--experiment-name` | in a terminal, the experiment `mlflow run` opens that run in; this notebook's kernel already names it |
| `python_env.yaml` | the environment the project needs: a Python version and its packages |
| environment manager | what prepares that environment: `local` uses the current one, `uv` and `virtualenv` build it |
""",
    )

    picture(
        "P6",
        "one command, the champion again",
        "`mlflow run` opens the run, fills in the entry point's command and starts it; given the champion's config, out comes the champion's score.",
        r"""
flowchart LR
    FILES["MLproject · python_env.yaml<br/>entry points: train, retrain"]
    CMD["!mlflow run . -e retrain<br/>-P config=runs:/…/config.json<br/>--env-manager local"]
    RUN[("run opened by MLflow<br/>source type PROJECT · entry point retrain")]
    TRAIN["python train.py --config runs:/…/config.json<br/>--run-name project-retrain"]
    SAME(["test_roc_auc ⟪pk_rebuilt_test⟫<br/>the champion's score, again"])
    TYPO{{"-P learning_rate=0.o5<br/>MLflow passes it on as typed"}}
    STOP(["train.py refuses it<br/>run FAILED"])
    FILES --> CMD --> RUN --> TRAIN --> SAME
    FILES -.-> TYPO --> STOP
""",
        r"""
    class FILES,CMD,TRAIN code
    class RUN stored
    class SAME endpoint
    class TYPO,STOP guard
""",
        r"""
- 🟦 files and commands · 🟪 stored in MLflow · 🟩 the result · 🟥 what `MLproject` doesn't check.
- The champion's run ID is all the command needs: the configuration comes from that run.
""",
    )

    md(
        "### `MLproject` — the whole file\n\n"
        "**Two named commands: `train` takes a configuration as parameters, `retrain` takes it from a run.**\n\n"
        "```yaml\n" + MLPROJECT_SRC + "\n```\n\n"
        + r"""- **`train`** is `train.py` with its four flags as parameters; the defaults are today's model. **`retrain`** is P2's hand-off, `train.py --config`, as a command with a name.
- **`command`** is a template: `{max_depth}` is replaced by the parameter's value. `>-` is YAML for one long line written across several.
- **`type`** is a label for whoever reads the file. MLflow puts the value into the command as typed, and `train.py`'s argument parser is what checks it (tested below).
- **The experiment isn't named in the file.** Which experiment a run goes into is the caller's choice, made on the command line.
- The file is named exactly `MLproject`, with no extension.
"""
    )

    md(
        "### `python_env.yaml` — the environment, as a file\n\n"
        "**The project names its Python version and points at the pins this notebook already installs.**\n\n"
        "```yaml\n" + PYTHON_ENV_SRC + "\n```\n\n"
        + r"""- `-r requirements.txt` instead of a second list of packages: one set of pins, so the project and the notebook can't drift apart.
- **`--env-manager` decides what happens with this file.** `local` leaves it alone and runs in the current environment, which is what every `mlflow run` in this notebook does. `uv` and `virtualenv` build the environment first (the end of this Part).
"""
    )

    md(
        r"""
### The first project run

**`mlflow run <project> -e <entry point> -P <name>=<value>`: MLflow opens a run, fills in the command and starts it.**

- `.` is the project: this folder. `-e train` picks the entry point. `-P max_depth=8` sets one parameter, and the other three keep their defaults.
- `--env-manager local`: run in this kernel's environment, without building one.
- **Which store and which experiment?** The ones this kernel's environment names. In P0, `set_tracking_uri` exported `MLFLOW_TRACKING_URI` and `set_experiment` exported `MLFLOW_EXPERIMENT_ID`; a `!` line inherits both. A terminal has neither (the end of this Part).
"""
    )

    code(
        r"""
# The `train` entry point with one parameter changed: MLflow opens the run, then starts train.py inside it
!mlflow run . -e train -P max_depth=8 --env-manager local
"""
    )

    md(
        r"""
- **`Running command`** shows the template filled in: `--max-depth 8` from `-P`, the other three values from their defaults.
- Depth 8 scores ⟪pk_depth8_test⟫ on test, between today's ⟪p1_test⟫ and the tuned ⟪p2_tuned_test⟫.
- MLflow logged the four parameters on the run itself, before `train.py` started. The run is named `project-train` because the command says so.

### Rebuild the champion

**A registered version leads to its run, the run holds `config.json`, and `retrain` turns that file into a model again.**

- `train.py` logs the configuration it used as `config.json` on every run (P1), so every version carries its own recipe.
- `champion` is looked up again here: since P5, `@champion` names version 2.
"""
    )

    code(
        r"""
# Rebuild the champion from its own run: that run's config.json, through the `retrain` entry point
champion = client.get_model_version_by_alias(REGISTERED_MODEL, "champion")  # version 2 since P5
champion_run_id = champion.run_id                                           # the run that trained it: xgb-tuned
print(f"@champion: version {champion.version} · run {champion_run_id}")

!mlflow run . -e retrain -P config=runs:/$champion_run_id/config.json --env-manager local
"""
    )

    code(
        r"""
# Every train.py run so far: the two project runs record how they were started
mlflow.search_runs(
    experiment_names=["booking-cancellations"],
    filter_string="tags.stage = 'pipeline'",
    order_by=["attributes.start_time ASC"],
)[["tags.mlflow.runName", "tags.mlflow.source.type", "tags.mlflow.project.entryPoint",
   "params.max_depth", "metrics.test_roc_auc", "tags.config_source"]]
"""
    )

    md(
        r"""
**Same configuration, same data, same code: the same model. `project-retrain` scores ⟪pk_rebuilt_test⟫, exactly what `xgb-tuned` scored.**

- Whoever ran it needed one thing: the champion's run ID. The configuration came from the run, the command from `MLproject`.
- The two project runs say how they were started: `mlflow.source.type` is `PROJECT`, and `mlflow.project.entryPoint` names the command. The `!python train.py` runs say `LOCAL`.
- A project run also records where its code came from, in `mlflow.source.name` and, inside a Git repository, `mlflow.source.git.commit`. Source, commit, entry point and parameters are what it takes to start the same run again.
- The registry still holds two versions. A project run logs a model; registering it stays a decision (P3).

### Who checks the parameters

**`MLproject` says `learning_rate` is a float. Send it a typo: the letter `o` for a zero.**
"""
    )

    code(
        r"""
# A typo in a parameter, o for 0: MLproject calls learning_rate a float, so who notices?
!mlflow run . -e train -P learning_rate=0.o5 --env-manager local
"""
    )

    md(
        r"""
**`train.py` caught it, not MLflow: `float` and `int` in `MLproject` are labels.**

- MLflow put `0.o5` into the command as typed. `argparse` refused it because `train.py` declares `--learning-rate` with `type=float`; a script that read its arguments loosely would have trained on whatever arrived.
- The same goes for a misspelled name: `-P max_dpeth=8` isn't refused, it's appended to the command as `--max_dpeth 8`, and `argparse` stops that too.
- What MLflow does check: that every parameter without a default is given.
- The failed run stays in the store with `learning_rate = 0.o5`: MLflow logs each parameter as typed, before the script starts.
- **Write floats as floats.** `-P min_child_weight=2` fails with *Changing param values is not allowed*: MLflow logged `2`, then `train.py` logged the value it parsed, `2.0`, and a logged parameter can't change. `-P min_child_weight=2.0` works.
"""
    )

    md(
        r"""
### A fresh environment, and a commit from GitHub — in a terminal

**`--env-manager local` trusts the environment you happen to be in. `uv` builds the one `python_env.yaml` describes, and a Git URL takes the code from a commit instead of this folder.**

```bash
cd "<the folder that contains this notebook>"
pip install uv        # once: MLflow calls uv to build the environment
MLFLOW_TRACKING_URI=sqlite:///mlflow.db mlflow run . -e train \
  --experiment-name booking-cancellations --env-manager uv
```

- **A terminal names the store and the experiment itself.** Leave `--experiment-name` out and the run opens in `Default`; `train.py` selects `booking-cancellations`, and MLflow stops it: *Cannot start run with ID … because active experiment ID does not match environment run ID.*
- In a `!` line of this notebook the flag isn't needed, and isn't accepted: `MLFLOW_EXPERIMENT_ID` is already set, and MLflow takes one of the two.
- ⟪pk_uv_line⟫
- It trains today's model again, test ROC-AUC ⟪p1_test⟫, in an environment nobody set up by hand. The run carries one more tag: `mlflow.project.env = uv`.
- With no `--env-manager` at all, MLflow picks `virtualenv` for this project: the same idea with `venv` and pip, and it needs pyenv to install the Python version.

```bash
MLFLOW_TRACKING_URI="sqlite:///$PWD/mlflow.db" mlflow run \
  "https://github.com/<owner>/<repository>#<folder>" --version <commit> \
  -e train --experiment-name booking-cancellations --env-manager uv
```

- Once this folder is in a Git repository, a URL takes the place of the `.`: MLflow clones the repository into a temporary folder, checks out `--version` (a commit or a branch) and runs the entry point there. `#<folder>` is the project's folder inside the repository.
- The tracking URI has to be absolute for this one. From the temporary folder, `sqlite:///mlflow.db` would be a different, empty file.
- The run records the URL and the commit, so "which code trained this model?" has an exact answer.
"""
    )

    md(
        r"""
### Where Projects fit

**`MLproject` is the smallest way to make a training job something other people, and machines, can start.**

- It writes down four things any production setup needs: named commands, declared parameters, an environment built from a file, and a pinned version of the code.
- Many teams get the same four from a container image that a CI job or a scheduler starts, and never write an `MLproject`. The file earns its place where the job should run with nothing but MLflow installed: a teammate's laptop, a notebook, a rerun of an old commit.
- The two meet: `MLproject` can name a Docker image (`docker_env`) in place of `python_env.yaml`, and `mlflow run` then starts the command inside a container.

💬 **Think about it** — `project-retrain` reproduced the champion's score exactly. What had to stay the same for that to happen, and which of those things does the run record?

✍️ **Try it** — Rebuild version 1 the same way, and check that it scores ⟪p1_test⟫ again.

<details><summary>Answer</summary>

```python
version_1_run_id = v1.run_id  # P3's v1: the version registered from xgb-depth6
!mlflow run . -e retrain -P config=runs:/$version_1_run_id/config.json --env-manager local
```

`test_roc_auc = ⟪p1_test⟫`. Version 1's `config.json` holds `train.py`'s defaults, so `-e train` with no `-P` builds the same model.

</details>
"""
    )

# ---------------------------------------------------------------- P6 : automate (Airflow folder only)

WEEKLY_TASKS_1_SRC = r'''
# Tasks 1–2: is there a week to process, and how did the champion do on it?
@task.short_circuit
def find_outcomes():
    """This run's week: data/outcomes/<logical date>.csv. No file yet → False, and every later task is skipped."""
    ds = get_current_context()["ds"]  # the run's logical date as YYYY-MM-DD: a Friday
    path = f"data/outcomes/{ds}.csv"
    if not (PROJECT / path).exists():
        print(f"find_outcomes: no {path} yet, nothing to do this week")
        return False
    print(f"find_outcomes: {path}")
    return path  # anything truthy lets the run go on, and reaches the next tasks through XCom


@task
def check_live(outcomes_path):
    """The champion on the week it served, now that the outcomes are known: logged as a run of its own."""
    week = pd.read_csv(PROJECT / outcomes_path)
    champion = client.get_model_version_by_alias(REGISTERED_MODEL, "champion")
    model = mlflow.pyfunc.load_model(f"models:/{REGISTERED_MODEL}/{champion.version}")
    p_cancel = model.predict(week[FEATURES].astype({c: "float64" for c in NUMERIC}))[:, 1]
    reconfirm = p_cancel >= THRESHOLD
    live = {
        "live_roc_auc": roc_auc_score(week.is_canceled, p_cancel),
        "reconfirm_calls": float(reconfirm.sum()),
        "reconfirm_precision": week.is_canceled[reconfirm].mean(),  # of the bookings called, the share that did cancel
    }
    mlflow.set_experiment("booking-cancellations")
    with mlflow.start_run(run_name=f"live-{get_current_context()['ds']}"):
        mlflow.set_tags({"stage": "live", "model_version": champion.version})
        mlflow.log_metrics(live)
    print(f"check_live: version {champion.version}, live ROC-AUC {live['live_roc_auc']:.4f}, "
          f"{live['reconfirm_calls']:.0f} reconfirm calls, {live['reconfirm_precision']:.0%} of them cancelled")
    return live
'''

WEEKLY_TASKS_2_SRC = r'''
# Tasks 3–4: retrain the champion's configuration with the week added, register the result as @challenger
@task.bash(cwd=str(PROJECT))
def retrain(outcomes_path):
    """The command to run: train.py with the champion's own config.json, plus the week's outcomes."""
    champion = client.get_model_version_by_alias(REGISTERED_MODEL, "champion")
    ds = get_current_context()["ds"]
    return (f"python train.py --config runs:/{champion.run_id}/config.json "
            f"--extra-train {outcomes_path} --run-name weekly-{ds}")  # its last line, model_uri = …, is the XCom


@task
def register(train_output):
    """Register the model train.py just logged as the next version, with the @challenger role."""
    model_uri = train_output.split()[-1]  # train.py's last line: "model_uri    = models:/m-…"
    context = get_current_context()
    version = mlflow.register_model(model_uri, REGISTERED_MODEL).version
    client.set_registered_model_alias(REGISTERED_MODEL, "challenger", version)
    client.update_model_version(
        REGISTERED_MODEL, version,
        description=f"Weekly retrain for {context['ds']}: the champion's configuration, plus that week's outcomes.",
    )
    client.set_model_version_tag(REGISTERED_MODEL, version, "airflow_run_id", context["run_id"])  # the DAG run that made it
    print(f"register: version {version} is @challenger")
    return version
'''

WEEKLY_TASKS_3_SRC = r'''
# Tasks 5–6: P5's gate, then P5's promotion; when the gate says no, promote is skipped
@task.short_circuit
def passes_gate(version):
    """P5's rule: at least MIN_GAIN more test ROC-AUC than the champion, under the size budget."""
    champion = client.get_model_version_by_alias(REGISTERED_MODEL, "champion")
    champion_metrics = mlflow.get_run(champion.run_id).data.metrics
    challenger_metrics = mlflow.get_run(client.get_model_version(REGISTERED_MODEL, version).run_id).data.metrics
    gain = challenger_metrics["test_roc_auc"] - champion_metrics["test_roc_auc"]
    passed = gain >= MIN_GAIN and challenger_metrics["model_size_mb"] < SIZE_BUDGET_MB
    print(f"passes_gate: version {version} vs {champion.version}, test ROC-AUC {gain:+.4f}, "
          f"{challenger_metrics['model_size_mb']:.1f} MB → {'promote' if passed else 'keep the champion'}")
    return passed


@task
def promote(version):
    """Move @champion to the version that passed, retire @challenger, record what it replaced."""
    champion = client.get_model_version_by_alias(REGISTERED_MODEL, "champion")
    client.set_registered_model_alias(REGISTERED_MODEL, "champion", version)
    client.delete_registered_model_alias(REGISTERED_MODEL, "challenger")
    client.set_model_version_tag(REGISTERED_MODEL, version, "promoted_over", f"version {champion.version}")
    print(f"promote: @champion is now version {version} (was {champion.version})")
'''

WEEKLY_DAG_SRC = r'''
# The DAG: when it runs, and the order of the six tasks. Passing one task's result to another sets the order
@dag(
    schedule="0 6 * * 5",                                # cron: Fridays at 06:00, once the week's file has landed
    start_date=pendulum.datetime(2017, 9, 1, tz="UTC"),  # the first Friday it may run for
    catchup=False,                                       # no runs for Fridays that have already gone by
    default_args={"retries": 1, "retry_delay": pendulum.duration(minutes=10)},  # a failed task gets one more try
    tags=["maré", "mlflow"],
)
def weekly_retrain():
    outcomes_path = find_outcomes()
    check_live(outcomes_path)
    version = register(retrain(outcomes_path))
    passes_gate(version) >> promote(version)  # >> : promote runs after the gate, and only if it passed


weekly_retrain_dag = weekly_retrain()  # the DAG object; in dags/weekly_retrain.py this line is what Airflow finds
'''

if AIRFLOW:
    part(
        "P6",
        "Automate: the weekly loop as an Airflow DAG",
        "run P5's loop every Friday on the week's outcomes, with no one at the keyboard",
        r"""
**The week P5 scored has happened, and its outcomes are in: who cancelled. From now on the same loop runs every Friday: collect the outcomes, check the champion, retrain, register, gate, promote. Airflow runs it; nobody re-runs cells.**

- One DAG, `weekly_retrain`, six tasks. Every task is code this notebook already ran: `train.py` from P2, the registry calls from P3, the gate from P5.
- Airflow runs it on a schedule, retries a task that fails, keeps every run's log, and shows all of it in a UI.
- **What this Part builds:** Airflow's folder beside this notebook, the six tasks and the DAG (the code of `dags/weekly_retrain.py`), a run for Friday 1 September and one for Friday 8 September.

| New word | What it means here |
|---|---|
| Airflow | a scheduler for batch work: it runs Python on a timetable, in order, and records every run |
| DAG | one workflow: tasks and the order between them (a directed acyclic graph) |
| task | one step of a DAG; here each is a Python function marked `@task` |
| `@dag` / `@task` | Airflow's decorators: a function becomes a DAG, or a task |
| schedule | when runs happen: `"0 6 * * 5"` is cron for Fridays at 06:00 |
| logical date | the date a run is for; tasks read it (`ds`) to pick their week of data |
| XCom | how one task's return value reaches the next task |
| `@task.short_circuit` | a task whose `False` skips everything after it: a clean "nothing to do", not a failure |
| `@task.bash` | a task that runs the shell command its function returns |
| `dag.test()` | runs one DAG run in this process, task by task: no scheduler, no server |
| `AIRFLOW_HOME` | Airflow's folder: its settings, its database of runs, and each task's log |
""",
    )

    picture(
        "P6",
        "one Friday, six tasks",
        "The week's file starts the run; two short-circuits decide how far it goes.",
        r"""
flowchart LR
    FIND{{"find_outcomes<br/>data/outcomes/2017-09-01.csv"}}
    LIVE["check_live<br/>@champion on the week it served<br/>live ROC-AUC ⟪p6_live_auc⟫"]
    TRAIN["retrain<br/>python train.py --config …<br/>--extra-train …"]
    REG[("register<br/>version 3 @challenger")]
    GATE{{"passes_gate<br/>gain ⟪p6_gain⟫, needs +0.002"}}
    PROMO["promote<br/>⟪p6_promote_label⟫"]
    FIND --> LIVE
    FIND --> TRAIN --> REG --> GATE --> PROMO
""",
        r"""
    class FIND,GATE guard
    class LIVE,TRAIN,PROMO code
    class REG stored
""",
        r"""
- 🟥 the two short-circuits: `False` skips everything after them · 🟦 tasks · 🟪 stored in MLflow.
- Friday 8 September has no outcomes file yet: `find_outcomes` returns `False` and the other five are skipped.
""",
    )

    md(
        r"""
### The week's outcomes

**The booking system drops one file per week: the week's arrivals with `is_canceled` filled in.**

- The same 1,090 bookings as P4 and P5's `upcoming`, in the same order, now with what happened.
- Named after the Friday it lands: `data/outcomes/2017-09-01.csv` covers arrivals from 25 to 31 August.
"""
    )

    code(
        r"""
# The week P5 scored, now with outcomes: one file per Friday in data/outcomes/
week = pd.read_csv("data/outcomes/2017-09-01.csv")

print(f"{len(week):,} bookings · arrivals {week.arrival_date.min()} → {week.arrival_date.max()} · cancelled {week.is_canceled.mean():.1%}")
print("the same bookings as P5's upcoming, in the same order:", (week.booking_id == upcoming.booking_id).all())
week[["booking_id", "arrival_date", "hotel", "lead_time", "market_segment", "deposit_type", "is_canceled"]].head(3)
"""
    )

    md(
        r"""
### Was P5's promotion right?

**P5 chose version 2 on test-set evidence. The week has happened, so the call can be checked against what guests did.**

- P5 scored these bookings with both versions: `p_champion` (version 1) and `p_challenger` (version 2), and found ⟪p5_flips⟫ decision flips.
"""
    )

    code(
        r"""
# Each version's decisions against the outcomes: reconfirm exactly the bookings that cancelled
cancelled = week.is_canceled.to_numpy() == 1
right_v1 = (p_champion >= THRESHOLD) == cancelled
right_v2 = (p_challenger >= THRESHOLD) == cancelled

print(f"ROC-AUC on the week : version 1 {roc_auc_score(cancelled, p_champion):.4f} · version 2 {roc_auc_score(cancelled, p_challenger):.4f}")
print(f"the {flips.sum()} flipped decisions: version 2 right on {right_v2[flips].sum()}, version 1 right on {right_v1[flips].sum()}")
"""
    )

    md(
        r"""
- ⟪p6_right_line⟫
- ⟪p6_noise_line⟫
- ⟪p6_oot_line⟫
- This check, and the retrain after it, has to happen every week. That's the job for a scheduler.
"""
    )

    md(
        r"""
### Airflow's folder

**Airflow keeps its settings, its record of every run and each task's log in one folder, `AIRFLOW_HOME`. Set it, and where the DAG files live, before Airflow is imported.**

- Without it, Airflow uses `~/airflow` in your home folder, shared by every project on the machine.
- The terminal command later points Airflow at the same folders, so the notebook and `airflow standalone` share one database of runs.
"""
    )

    code(
        r"""
# Airflow's folder and settings, beside this notebook: set before `import airflow`, inherited by `!airflow`
os.environ["AIRFLOW_HOME"] = os.path.abspath("airflow")             # settings, the database of runs, task logs
os.environ["AIRFLOW__CORE__DAGS_FOLDER"] = os.path.abspath("dags")  # where Airflow looks for DAG files
os.environ["AIRFLOW__CORE__LOAD_EXAMPLES"] = "False"                # no example DAGs in the UI
os.environ["AIRFLOW__LOGGING__LOGGING_LEVEL"] = "WARNING"           # the tasks' own prints, not Airflow's bookkeeping

!airflow db migrate && ls airflow
"""
    )

    code(
        r"""
# Airflow's decorators, pendulum for dates with a time zone, and the paths the tasks use
from pathlib import Path

import pendulum
from airflow.sdk import dag, get_current_context, task

PROJECT = Path.cwd()  # this folder: train.py, data/, mlflow.db (the DAG file works it out from its own path)
mlflow.set_tracking_uri(f"sqlite:///{PROJECT / 'mlflow.db'}")  # the same store, as an absolute path: tasks may run from any folder
"""
    )

    md(
        r"""
### The six tasks

**Each task is a plain function with a decorator. What one returns, the next one receives.**

- `find_outcomes` looks for the file named after the run's logical date. `check_live` scores the champion on it and logs the result as a run.
"""
    )

    code(WEEKLY_TASKS_1_SRC)

    md(
        r"""
- `retrain` is `@task.bash`: it returns the command, Airflow runs it in the project folder, and the command's last line (`model_uri = models:/m-…`) becomes its result.
- The command is P2's hand-off, with the champion's own `config.json` and the new flag `--extra-train`: the week's bookings join the training rows, and the test set stays the one every version was scored on.
"""
    )

    code(WEEKLY_TASKS_2_SRC)

    code(WEEKLY_TASKS_3_SRC)

    md(
        r"""
**The DAG itself is a function that calls the tasks in order.**

- Calling a task inside it doesn't run the task: it adds a step to the graph and wires its result to whatever receives it.
- `retries=1` gives a failed task one more try, 10 minutes later, before the run is marked failed.
"""
    )

    code(WEEKLY_DAG_SRC)

    md(
        r"""
### Run it for Friday 1 September

**`dag.test()` runs one DAG run in this process: the six tasks in order, each one's prints below.**

- Airflow records it like any other run, in `airflow/airflow.db`: the UI will show it.
"""
    )

    code(
        r"""
# One DAG run for Friday 1 September, in this process
friday = weekly_retrain_dag.test(logical_date=pendulum.datetime(2017, 9, 1, 6, tz="UTC"))

states = {ti.task_id: ti.state for ti in friday.get_task_instances()}
print(f"\n{friday.run_id} → {friday.state}")
{task_id: states[task_id] for task_id in weekly_retrain_dag.task_ids}  # in the DAG's order
"""
    )

    md(
        r"""
- ⟪p6_run_line_1⟫
- ⟪p6_run_line_2⟫
- The DAG run is a success either way: a skipped `promote` means the gate decided, not that anything broke.
"""
    )

    code(
        r"""
# What the Friday run left in MLflow: the live check and the retrain, as runs
mlflow.search_runs(
    experiment_names=["booking-cancellations"],
    filter_string="attributes.run_name LIKE '%-2017-09-01'",
    order_by=["attributes.start_time ASC"],
)[["tags.mlflow.runName", "tags.stage", "metrics.live_roc_auc", "metrics.reconfirm_precision",
   "metrics.test_roc_auc", "params.extra_train"]]
"""
    )

    code(
        r"""
# And in the registry: a third version, which DAG run made it, and where the aliases point now
aliases = client.get_registered_model(REGISTERED_MODEL).aliases
pd.DataFrame([
    {
        "version": mv.version,
        "aliases": [alias for alias, version in aliases.items() if int(version) == int(mv.version)],
        "airflow_run_id": mv.tags.get("airflow_run_id", ""),
        "description": mv.description,
    }
    for mv in client.search_model_versions(f"name = '{REGISTERED_MODEL}'")
]).sort_values("version")
"""
    )

    md(
        r"""
### Friday 8 September: nothing to do yet

**No outcomes file for that week: `find_outcomes` returns `False`, and Airflow skips the other five tasks.**
"""
    )

    code(
        r"""
# The next Friday: no data/outcomes/2017-09-08.csv, so the run stops after its first task
next_friday = weekly_retrain_dag.test(logical_date=pendulum.datetime(2017, 9, 8, 6, tz="UTC"))

states = {ti.task_id: ti.state for ti in next_friday.get_task_instances()}
print(f"\n{next_friday.run_id} → {next_friday.state}")
{task_id: states[task_id] for task_id in weekly_retrain_dag.task_ids}
"""
    )

    md(
        r"""
- Skipped, not failed: the run did its job, which was to find nothing to do. A failed task shows red, gets its retry, and waits for someone to read its log.
- Nothing new in MLflow: no live check, no training run, no version.
"""
    )

    md(
        r"""
### The Airflow UI — in a terminal

**`airflow standalone` starts every Airflow component in one terminal: the API server with the UI, the scheduler, the DAG processor and the triggerer.**

```bash
cd "<the folder that contains this notebook>"
export AIRFLOW_HOME="$PWD/airflow" AIRFLOW__CORE__DAGS_FOLDER="$PWD/dags"
export AIRFLOW__CORE__LOAD_EXAMPLES=False
airflow standalone
```

⟪p6_ui_steps⟫
"""
    )

    md(
        r"""
### Notebook → `dags/weekly_retrain.py`

**The DAG file holds exactly the code of P6's task and DAG cells, plus what they use.**

| In this notebook | In `dags/weekly_retrain.py` |
|---|---|
| `NUMERIC`, `CATEGORICAL`, `FEATURES` (P1), `REGISTERED_MODEL` (P3), `THRESHOLD` (P4), `MIN_GAIN`, `SIZE_BUDGET_MB` (P5) | the same constants, at the top |
| `client` (P0) | the same `MlflowClient()` |
| `PROJECT = Path.cwd()` and the tracking URI (P6) | `PROJECT` from the file's own path, then the same absolute tracking URI |
| the six tasks, `weekly_retrain()`, `weekly_retrain_dag` (P6) | identical |
| `weekly_retrain_dag.test(...)` | not there: the scheduler creates the runs |

💬 **Think about it** — ⟪p6_think⟫

✍️ **Try it** — Run the DAG for 1 September again with `MIN_GAIN = -1`, so the gate passes whatever the gain. Which version is `@champion` afterwards, and what puts version 2 back?

<details><summary>Answer</summary>

```python
MIN_GAIN = -1  # passes_gate reads it when it runs
weekly_retrain_dag.test(logical_date=pendulum.datetime(2017, 9, 1, 6, tz="UTC"))
client.get_registered_model(REGISTERED_MODEL).aliases  # {'champion': 4}: new version, promoted

client.set_registered_model_alias(REGISTERED_MODEL, "champion", 2)  # roll back, then restart
MIN_GAIN = 0.002
```

Airflow keeps one run per logical date, so the new run replaces the first 1 September run in its UI. MLflow keeps both: versions 3 and 4, each tagged with the DAG run that made it.

</details>
"""
    )

    md(
        r"""
### Where this runs for real

**The DAG file stays the same; what's around it grows.**

- A server runs the scheduler and the workers, with Postgres in place of SQLite.
- The outcomes arrive from the booking system: a bucket or a warehouse table, with a sensor task waiting for them.
- A task that fails its retry alerts someone (email, Slack), with the task's log one click away.
- One more task after `promote` would restart the two endpoints, so a new champion goes live without a person.
"""
    )

# ---------------------------------------------------------------- wrap-up

part(
    WRAP,
    "Wrap-up",
    "the whole path on one page, and what builds on it",
    (
        r"""
### Tracking, registry, serving, orchestration

**Each answers a different question, and changes in a different way when the model does.**

| | Experiment tracking | Model Registry | Serving | Orchestration |
|---|---|---|---|---|
| **Question** | what did we try, and how did it do? | which model is approved, and what did it replace? | how do other systems call it? | who runs the loop, when, and what happened each time? |
| **Unit** | run → params, metrics, artifacts, logged model | registered model → versions → aliases | an endpoint answering from one version | DAG → runs → tasks |
| **Here** | P1–P2: `train.py`, `tune-xgb`, `xgb-tuned` | P3, P5: `booking-cancellation`, `@champion` | P4: `/invocations`, `/predict`, `/health` | P6: `weekly_retrain`, Fridays 06:00 |
| **When a better model arrives** | a new run | an alias moves | a restart | the Friday run trains, gates and promotes it |
"""
        if AIRFLOW else
        r"""
### Tracking, registry, serving, projects

**Each answers a different question, and changes in a different way when the model does.**

| | Experiment tracking | Model Registry | Serving | Projects |
|---|---|---|---|---|
| **Question** | what did we try, and how did it do? | which model is approved, and what did it replace? | how do other systems call it? | how does anyone else run the training? |
| **Unit** | run → params, metrics, artifacts, logged model | registered model → versions → aliases | an endpoint answering from one version | project → entry points → parameters |
| **Here** | P1–P2: `train.py`, `tune-xgb`, `xgb-tuned` | P3, P5: `booking-cancellation`, `@champion` | P4: `/invocations`, `/predict`, `/health` | P6: `MLproject`, `mlflow run . -e retrain` |
| **When a better model arrives** | a new run | an alias moves | a restart | nothing changes: the same command trains the next one |
"""
    ),
)

md(
    r"""
### The workflow, end to end

**About ten calls cover this notebook.**

```python
with mlflow.start_run(run_name="tune-xgb"):                           # P2  the parent
    study.optimize(objective, n_trials=20)                            #     trials: nested=True
    mlflow.log_dict(best_config, "best_config.json")                  #     the answer, typed
!python train.py --config runs:/<tuning_run_id>/best_config.json      # P2  the challenger
mlflow.register_model("models:/m-…", REGISTERED_MODEL)                # P3  → version N
client.set_registered_model_alias(REGISTERED_MODEL, "challenger", N)  # P3  a role
mlflow.pyfunc.load_model(f"models:/{REGISTERED_MODEL}@champion")      # P3  every consumer
client.get_model_version_by_alias(REGISTERED_MODEL, "champion")       # P4  which version, once
client.set_registered_model_alias(REGISTERED_MODEL, "champion", N)    # P5  promote / roll back
"""
    + (
        r"""weekly_retrain_dag.test(logical_date=...)                             # P6  one run, in-process
"""
        if AIRFLOW else ""
    )
    + r"""```

```bash
mlflow models serve -m "models:/booking-cancellation@champion" --env-manager local --port 5002
uvicorn app:app --port 8001
"""
    + ("airflow standalone\n" if AIRFLOW else
       "mlflow run . -e retrain -P config=runs:/<run_id>/config.json \\\n"
       "  --experiment-name booking-cancellations --env-manager local\n")
    + r"""```
"""
)

md(
    r"""
### Recommendations for real projects

**Small rules that keep a model replaceable.**

- **Consumers load aliases, never versions or run IDs.** `models:/<name>@champion` is the only model reference in `app.py`.
- **Resolve the alias once per process**, then load that version, so the answer and the reported version always match.
- **Put the promotion gate in code**, next to the numbers it reads, and record each promotion as a version tag.
- **Know what a promotion changes:** score recent traffic with both versions before moving the alias.
- **A signature is not a contract for values.** Put a schema in front of any model that takes requests from outside the team.
- **Hand configurations over as artifacts** (`runs:/…/best_config.json`), never as numbers copied between cells or scripts.
- **One parent run per search**, a child per attempt, and the answer logged on the parent.
- **Aliases, not stages:** stages are deprecated, and an alias can mean anything your process needs (`@champion`, `@challenger`, `@shadow`).
"""
    + (
        r"""- **Let a schedule run the loop, not a person.** The DAG does the same steps every week, and each run leaves its record in Airflow and in MLflow.
- **Keep DAG tasks thin:** they call `train.py` and the same registry calls you'd make by hand, so the DAG never becomes a second copy of the training code.
"""
        if AIRFLOW else
        r"""- **Give the training job a named command** with declared parameters and an environment file (`MLproject`), so a teammate, CI or a scheduler starts it exactly the way you do.
- **Log the configuration on every run** (`config.json`): any version can then be rebuilt from its own run.
"""
    )
)

md(
    r"""
### What comes next

**Everything below starts from `models:/booking-cancellation@champion`.**

- **Containers** — `mlflow models build-docker -m "models:/booking-cancellation@champion" -n mare-cancel` builds an image of the scoring server with the model inside; `app.py` gets its own Dockerfile.
"""
    + (
        r"""- **Airflow on a server** — the scheduler, workers and Postgres run somewhere that stays up; the same DAG file, now with alerts and a deploy task after `promote`.
"""
        if AIRFLOW else
        r"""- **Automation** — CI or a scheduler runs the `retrain` entry point and the gate on every candidate, and retrains as new bookings arrive.
"""
    )
    + r"""- **A shared tracking server** — one `MLFLOW_TRACKING_URI` for the team, a database and object storage behind it, and permissions on who may move `@champion`.
- **Monitoring** — the inputs `/predict` receives compared with the training data (the `lead_time`-in-weeks mistake), and outcomes as bookings arrive or cancel.
"""
    + (
        r"""- **MLflow Projects** — an `MLproject` file plus `mlflow run` package a training command with its environment. A container started by a CI job does the same work, and it's what most teams use.
"""
        if AIRFLOW else ""
    )
)


# ------------------------------------------------------------------ app.py / DAG checks

def top_level(source):
    """name → AST dump of every top-level class, function and simple assignment in `source`."""
    found = {}
    for node in ast.parse(source).body:
        if isinstance(node, (ast.ClassDef, ast.FunctionDef)):
            found[node.name] = ast.dump(node)
        elif isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            found[node.targets[0].id] = ast.dump(node.value)
    return found


def compare(path, may_differ=()):
    """Every name the file shares with the notebook's code cells must have identical code in both."""
    notebook = {}
    for cell in cells:
        src = "".join(cell["source"])
        magic = any(line.startswith(("%", "!")) for line in src.splitlines())  # not plain Python: skip
        if cell["cell_type"] == "code" and not magic:
            notebook.update(top_level(src))
    in_file = top_level(path.read_text())
    shared = sorted(set(in_file) & set(notebook))
    differ = [name for name in shared if in_file[name] != notebook[name] and name not in may_differ]
    missing = sorted(set(in_file) - set(notebook))
    return shared, differ, missing


def report(label, path, may_differ=()):
    shared, differ, missing = compare(path, may_differ)
    print(f"{label} vs notebook: {len(shared)} shared names, "
          + ("✅ identical" if not differ else f"❌ different: {differ}")
          + (f" (by design: {', '.join(may_differ)})" if may_differ else "")
          + (f" · only in the file: {missing}" if missing else ""))


def report_project():
    """MLproject against train.py and the notebook: flags, placeholders, and the names the cells run."""
    try:
        import yaml  # comes with MLflow; only this check needs it
    except ImportError:
        print("MLproject vs train.py: not checked (PyYAML isn't installed here)")
        return
    entry_points = yaml.safe_load((HERE / "MLproject").read_text())["entry_points"]
    flags = set(re.findall(r'add_argument\("(--[a-z-]+)"', (HERE / "train.py").read_text()))
    problems = []
    for name, entry in entry_points.items():
        declared = set(entry.get("parameters", {}))
        placeholders = set(re.findall(r"\{(\w+)\}", entry["command"]))
        unknown = sorted(set(re.findall(r"--[a-z][a-z-]*", entry["command"])) - flags)
        if unknown:
            problems.append(f"{name}: train.py has no {', '.join(unknown)}")
        if placeholders != declared:
            problems.append(f"{name}: parameters and placeholders differ on {sorted(placeholders ^ declared)}")
    for cell in cells:  # every `mlflow run . -e <entry point> -P <parameter>=…` a code cell runs
        src = "".join(cell["source"]).replace("\\\n", " ")
        for used, rest in re.findall(r"mlflow run \. -e (\w+)(.*)", src if cell["cell_type"] == "code" else ""):
            if used not in entry_points:
                problems.append(f"a cell runs entry point {used}, which MLproject doesn't have")
                continue
            for parameter in re.findall(r"-P (\w+)=", rest):
                if parameter not in entry_points[used].get("parameters", {}):
                    problems.append(f"a cell passes {parameter} to {used}, which doesn't declare it")
    print(f"MLproject vs train.py and the notebook: {len(entry_points)} entry points, "
          + ("✅ every flag, placeholder and parameter matches" if not problems else f"❌ {'; '.join(problems)}"))


# ------------------------------------------------------------------ fill in the measured numbers

def fill(text):
    return re.sub(r"⟪(\w+)⟫", lambda m: str(MEASURED[m.group(1)]), text)


for cell in cells:
    cell["source"] = fill("".join(cell["source"])).splitlines(keepends=True)

# ------------------------------------------------------------------ emit

nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": KERNELSPEC,
        "language_info": {"name": "python", "version": "3.12.12"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None)
    out_dir = Path(ap.parse_args().out or HERE)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / NB_NAME
    path.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n")
    n_code = sum(1 for c in cells if c["cell_type"] == "code")
    print(f"{path}  ({len(cells)} cells: {n_code} code, {len(cells) - n_code} markdown)")
    report("app.py", HERE / "app.py")
    if AIRFLOW:
        report("dags/weekly_retrain.py", HERE / "dags" / "weekly_retrain.py", may_differ=("PROJECT",))
    else:
        report_project()
    unmeasured = sorted({k for k, v in MEASURED.items() if v == "TBD"})
    if unmeasured:
        print(f"⚠️  MEASURED still TBD: {', '.join(unmeasured)}")
