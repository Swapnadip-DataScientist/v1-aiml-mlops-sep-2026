# lec 6 — MLflow: tune, register, serve · version log

| Version | Notebook | Cells | Execution | Built from |
|---|---|---|---|---|
| v1 (2026-09-30), HistGradientBoosting | `v1_MLflow_Tuning_Registry_Serving.ipynb` | 72 (29 code, 43 markdown), 0.16 MB | ~139 s top to bottom on dev3.12, 0 errors | `_build_nb.py` |
| v1 (2026-09-30), **XGBoost** | the same file, rebuilt | 71 (29 code, 42 markdown), 0.16 MB | ~60 s top to bottom on dev3.12, 0 errors | the script as it was before v2; kept in the folder as executed |
| v2 (2026-09-30), XGBoost + **P6 · Package** (current) | `v2_MLflow_Tuning_Registry_Serving.ipynb` | 85 (33 code, 52 markdown), 0.18 MB | P0–P5 keep v1's outputs; the 4 new cells ran against the same store in ~35 s. A top-to-bottom run from an empty store: ~100 s, 0 errors | `_build_nb.py`, shared with the Airflow folder |

**Renamed 2026-09-30, in v2:** the variable `MODEL_NAME` is now `REGISTERED_MODEL`, in the notebook,
`app.py` and the build script. "Model name" read as the algorithm (XGBoost); the variable holds the
registered model's name in the registry. P3 also gained one bullet: the name says what the model is
for, not which algorithm it is. Sources only; no cell was re-run and no output changed. v1 keeps
`MODEL_NAME` in its own cells and still runs, because it defines that variable itself.

`_build_nb.py` is the master copy, and it now builds **v2**. The same file builds the Airflow variant:
a folder that has `dags/` gets the Airflow Part (P6: automate) and its notebook name; this folder has
none, so it gets the MLflow Projects Part (P6: package). `MEASURED`, at its top, holds every number
the prose quotes; each build fills them in and refuses a missing one. `train.py`, `app.py`,
`MLproject`, `python_env.yaml`, `requirements.txt` and `README.md` are hand-maintained files beside
it. `data/` is a byte-identical copy of lec 5's `data/`.

```bash
python _build_nb.py                    # writes the notebook next to this file
python _build_nb.py --out /tmp/x       # writes it elsewhere, for diffing first
```

Every build also compares `app.py` with the notebook's code cells (AST, name by name) and prints
`app.py vs notebook: 10 shared names, ✅ identical` or the names that drifted. For P6 it checks
`MLproject` too: every `--flag` in a command exists in `train.py`, every `{placeholder}` is a declared
parameter, and every entry point and `-P` name a code cell runs exists. It prints
`MLproject vs train.py and the notebook: 2 entry points, ✅ every flag, placeholder and parameter matches`.

**Moved 2026-09-30** from the lec 6 folder into `current version/`, next to `airflow version/`, with no
code change. The store's 30 absolute paths were rewritten then. That store was replaced when the
notebook was re-run with XGBoost; the HistGradientBoosting run is gone from this folder. The
README's row for the colleague's notebook was dropped, because that notebook is not in this folder.

---

## v1 → v2: P6 · Package, MLflow Projects (2026-09-30)

### Why

- Your colleague's Class6 notebook is mostly MLflow Projects, and v1 gave the subject one sentence in
  its wrap-up. You asked for it here: "MLproject what's that? we not covering it?", then "yea do it,
  in current_version" and "make the notebook v2_ don't disturb current notebook".
- Projects is the one part of the usual four-part description of MLflow (Tracking, Projects, Models,
  Registry) that neither lec 5 nor lec 6 showed.
- It takes this version from about 2 h to about 2 h 25 min.

### What changed

- **`v1_MLflow_Tuning_Registry_Serving.ipynb`: I didn't write to it.** It still has its 71 cells, its
  outputs and your one hand edit (below).
- **New files:** `MLproject` (entry points `train` and `retrain`) and `python_env.yaml`.
- **`v2_MLflow_Tuning_Registry_Serving.ipynb`:** v1's 71 cells plus 14. P6 is 14 cells (4 code,
  10 markdown) after P5; the wrap-up is now P7.
- **P0–P5's 29 code cells are identical to v1's**, and carry v1's outputs.
- **Your hand edit is folded in.** At 17:52 you added a framing question under P3's heading in v1:
  "Out of all these trained models, which model is actually the official model, which one is the
  challenger, and how should applications refer to them?". It's now in the build script
  (`part(..., question=…)`), word for word, so v2 has it. It applies to this folder only: the Airflow
  notebook doesn't have that line.
- **Markdown that changed outside P6:**
  - the title and intro (", package"; a P6 row; P7 for the wrap-up);
  - one bullet in "What gets built", and the map's P6 entry and END RESULT box;
  - every Part's strip ("Part n of 7", with `package`);
  - P0's list of files;
  - the wrap-up: a Projects column, the `mlflow run` line, two recommendations. "What comes next" loses
    its MLflow Projects bullet, and Automation now names the `retrain` entry point.
- **Unchanged:** `train.py`, `app.py`, `requirements.txt`, `data/`.
- **`README.md`:** the P6 row, the two commands, step 7 (the `uv` run), the new files, and v1 listed as
  "the notebook as it was before P6".

| P6 cell | Content |
|---|---|
| opening | what a project is, what the Part builds, 9 new words |
| picture | `mlflow run . -e retrain` → a PROJECT run → `train.py --config` → 0.9169 again; the typo path |
| `MLproject` | the whole file, quoted from disk, and what each key does |
| `python_env.yaml` | the whole file; what `--env-manager` does with it |
| the first project run | `!mlflow run . -e train -P max_depth=8 --env-manager local` → `project-train`, 0.9092 |
| rebuild the champion | `champion` looked up again (version 2), `!mlflow run . -e retrain -P config=runs:/…/config.json` → `project-retrain`, 0.9169 |
| the four `train.py` runs | source type, entry point, depth, test ROC-AUC, config source |
| who checks the parameters | `-P learning_rate=0.o5` → `argparse` refuses it, the run is FAILED |
| in a terminal | `--env-manager uv`; a commit from GitHub |
| where Projects fit | against a container + CI; Think about it; Try it (rebuild version 1 → 0.9017) |

### Decisions, and the measurements behind them (MLflow 3.16.1)

- **Two entry points.** `train` is `train.py`'s four flags as parameters, with today's model as the
  defaults. `retrain` is P2's hand-off, `--config`, as a named command. `retrain` with the champion's
  own `config.json` gives the Part its result: test ROC-AUC 0.9168645962828026 in both runs.
- **The run names are in the commands** (`--run-name project-train` / `project-retrain`).
  `train.py` calls `start_run(run_name=…)`, which renames the run MLflow opened, so
  `mlflow run --run-name` has no effect. Without this both runs would be `train-xgb`.
- **`config` is `type: string`, not `uri`.** A value that fails MLflow's `uri` check leaves the run in
  `RUNNING` for good: the run is created before the check. As a `string`, a bad value fails inside
  `train.py` and the run ends `FAILED`.
- **`float` and `int` aren't checked.** `0.o5` reaches `argparse`. A `-P` name the entry point doesn't
  declare is appended to the command as `--name value`. A missing required parameter is refused.
- **`mlflow run` logs the parameters as typed, before the script starts.** So
  `-P min_child_weight=2` fails with "Changing param values is not allowed" (`2`, then `train.py`'s
  `2.0`). The notebook says so in one bullet.
- **No `--experiment-name` in the notebook's cells.** `mlflow.set_experiment` exports
  `MLFLOW_EXPERIMENT_ID`, and the CLI refuses both: "Specify only one of 'experiment-name' or
  'experiment-id' options". My first draft had the flag in every cell; running the cells in a real
  kernel caught it. In a terminal, without the flag, the run opens in `Default` and `train.py` fails
  with "Cannot start run with ID … because active experiment ID does not match environment run ID".
- **That experiment error is described, not run.** Its traceback prints this machine's paths, and
  trimming it needs `| tail`, which a Windows shell doesn't have. `x = !cmd` isn't an option either:
  it's a SyntaxError on IPython 8.12.3 with Python 3.12. It's a bullet in the terminal section.
- **`--env-manager uv`, measured** with `MLFLOW_ENV_ROOT` pointed at a scratch folder:
  - first run 1 min 55 s with an empty uv cache, 55 s with a warm one; a second run 11 s;
  - 585–591 MB per environment (default location `~/.mlflow/envs/`);
  - it trains 0.9016508827889413, exactly `xgb-depth6`'s score, and the run gets
    `mlflow.project.env = uv`;
  - without `build_dependencies: pip`, `log_model` warns "Failed to resolve installed pip version".
- **`virtualenv`, the default for a `python_env` project, was not run.** It needs pyenv to install
  Python, and this Mac's pyenv has only `system`, so it would compile one. The notebook's one line about
  it comes from MLflow's source and `mlflow run --help`.
- **Running from Git, checked with MLflow's public example**
  (`mlflow run https://github.com/mlflow/mlflow-example.git --version <commit>`):
  - with a relative `sqlite:///…` URI the run fails, "Run with id=… not found": the clone sits in a
    temporary folder;
  - with an absolute URI it succeeds, and the run records the URL, the commit and the repository.
  This project itself wasn't run from Git: it isn't in a public repository.

### Deliberately left out

- `conda_env`, and `docker_env` beyond one bullet; the Databricks and Kubernetes backends.
- `mlflow.projects.run()`, the Python API; multi-step workflows, where an entry point starts others.
- The `path` and `uri` parameter types.
- A `tune` entry point: the search lives in the notebook, not in a script.
- The experiment-mismatch error as a live cell (see above).

### How it was built and executed

- `_build_nb.py` first. The Airflow folder's copy was updated too and is byte-identical; that folder's
  notebook still matches its build, 99 of 99 cells, and wasn't touched.
- **v2 was assembled, not re-run.** Its 29 shared code cells take v1's outputs, matched by source. The
  4 new code cells ran on their own against the existing store, after P0's setup cells and
  `MODEL_NAME`; their execution counts continue at 30–33.
- **A full run as well, in a scratch copy from an empty store:** 100.7 s, 0 errors. All 29 shared cells
  print what v1 prints, run IDs and timings aside. P6's numbers match the ones above, and the Try-it's
  answer gives 0.9017.
- P6's picture was rendered through mermaid.ink in light and dark themes.

### Caveats

- **v1 and v2 share this folder and its store.** The store now also holds P6's runs: `project-train`,
  `project-retrain` and one FAILED run (the typo). v1's runs and versions are untouched, so its outputs
  still match; re-running v1's "both train.py runs" cell would list four runs.
- The registry is as v1 left it: `@champion` → version 2, two versions, no `@challenger`.
- Delete `mlflow.db` and `mlruns/` before a live run of either notebook, as before.
- **`uv` isn't in `requirements.txt`.** The terminal step says `pip install uv`. Nothing was installed
  into dev3.12, and no environment was created under `~/.mlflow/`: the first `--env-manager uv` run on
  this machine will build one.
- Runs record the Git repository the folder sits in (repository URL and commit, as tags), as every
  `train.py` run already did. The folder is untracked there, so that commit doesn't contain these files.

### Pacing (kept here, not in the notebook)

85 cells. About 2 h 25 min at a steady pace with discussion:

| Parts | Time |
|---|---|
| P0 – P5 | ~1 h 55 min (as v1, below) |
| P6 (each `mlflow run` takes ~13 s; the `uv` run is a terminal step of a minute or two) | ~25 min |
| P7 | ~5 min |

---

## HistGradientBoosting → XGBoost (2026-09-30)

**Your call:** XGBoost is the model more teams run in production; HistGradientBoosting is rare. Both
folders switched. `train.py`, the notebook's pipeline and search, and every number changed; the
structure didn't.

| | HistGradientBoosting | XGBoost |
|---|---|---|
| today's model (`train.py` defaults) | 127 leaves, lr 0.1, `min_samples_leaf` 20, ≤ 300 iterations: 0.9139, 3.2 MB | 300 trees, depth 6, lr 0.1, `min_child_weight` 1: **0.9017**, 0.9 MB |
| run names | `hist_gb-127leaves`, `tune-hist_gb`, `hist_gb-tuned` | `xgb-depth6`, `tune-xgb`, `xgb-tuned` |
| search | lr 0.05–0.3, 31–1023 leaves, `min_samples_leaf` 1–100 | lr 0.02–0.3, `max_depth` 3–14, `min_child_weight` 1–20 (log where it helps) |
| 20 trials | ~90 s (4 threads) | ~30 s |
| best trial (validation) | trial-13, 0.9140; random-phase best 0.9126 | trial-13, 0.9134 (lr 0.089, depth 12, 1.03); random-phase best trial-08, 0.9114 |
| challenger (test) | 0.9192, 11.8 MB, gain +0.0053 | **0.9169**, 5.6 MB, gain **+0.0152** |
| BK218304 as sent / "Online Travel Agent" / `lead_time -1` / "city hotel" | 0.7929 / 0.3861 / 0.1408 / 0.8218 | 0.7255 / 0.3773 / 0.2004 / 0.7771 |
| next week: reconfirm calls, flips | 277 → 274, 57 (30 off, 27 on) | 268 → 271, **77** (37 off, 40 on) |
| BK218304 after promotion | 0.8330 | 0.7908 |

- **The thread note is gone.** XGBoost 3.2.0 fits in 0.5–1.9 s here at any thread count, with
  identical scores; depth 12 took 1.7 s on the default 16 threads and 1.1 s on 4. So
  `OMP_NUM_THREADS=4` left P0, its markdown cell and `train.py`.
- **XGBoost's library defaults (100 trees, lr 0.3) score 0.9005**, close to `train.py`'s defaults. Depth
  is what the search finds: the top four trials are 12–14 deep.
- **`subsample` / `colsample_bytree` stay out of the search:** on a grid they added at most 0.0006 on
  validation.
- **BK218304 is still P4's example.** Both "Online Travel Agent" and `lead_time = -1` flip its reconfirm
  call. The earlier text said one of the two did; with HistGradientBoosting it was also two.
  `lead_time = -1` scores exactly as 0 for every booking; same-day bookings cancel 6.8%.
- **P2's Try-it now filters on model size**, which is the same on every run, not fit time. The answer is
  `trial-18` and `trial-15`: depth 8, within 0.006 of the best at about a third of its size.
- **On a Mac, XGBoost needs the OpenMP runtime.** Its library links `@rpath/libomp.dylib`, searched in
  `/opt/homebrew/opt/libomp/lib`, so a Mac needs `brew install libomp`. The notebook's P0 and the
  README say so. It loads alongside scikit-learn's own `libomp` without trouble.
- **New pins:**
  - `xgboost` 3.2.0, `numpy` 2.3.3 and `scipy` 1.17.0, so both folders print the same numbers.
  - `httpx2` 2.3.0: without it, Starlette's TestClient warns on import.
  - `ipywidgets` 8.1.8: without it, tqdm warns on import.
  - All five were already in dev3.12, and a dry run of the install cell there installs nothing.

---

## Class6 → v1

### Why

- `Class6_MLflow_Part2_Projects_and_Models.ipynb` (your colleague's, left untouched one level up) is
  the second half of a different track. It continues the colleague's term-deposit `campaign/` repo,
  Class5's skops trust list and Class4's FastAPI service, none of which our lec 4 (Milepost) or lec 5
  (Maré Hotels) built.
- By cell count:
  - about 45% of it is MLflow Projects machinery and its setup (`MLproject`, `mlflow run`,
    `--env-manager uv`, running a commit from GitHub, the experiment-name clash), a feature the notebook
    itself says most teams skip;
  - about 25% repeats lec 5 (autolog, the model folder and flavors, the int-to-float signature refusal);
  - about 25% was new and worth keeping: nested runs for a search, "what the signature lets through",
    and `mlflow models serve` against a hand-built FastAPI service.
- Agreed scope (2026-09-30): continue Maré Hotels in four Parts: tune, register, serve and guard,
  promote. `MLproject` gets one sentence in the wrap-up.

### What v1 contains

| Part | Content |
|---|---|
| P0 | pinned install (with the `brew install libomp` note), autotime, imports, tracking URI + experiment, `mlflow ui` command |
| P1 | data + codes table, the split, `train.py`'s key lines, `!python train.py --run-name xgb-depth6` → test ROC-AUC 0.9017, the run's logged model via `run.outputs.model_outputs` |
| P2 | `build_pipeline` / `evaluate` (same as `train.py`); `objective()` with `start_run(nested=True)`; parent `tune-xgb` (20 trials, TPE seed 42, `best_config.json`, search-history plot); leaderboard via `tags.mlflow.parentRunId`; `train.py --config runs:/…/best_config.json` → `xgb-tuned`, test ROC-AUC 0.9169 |
| P3 | `register_model(models:/m-…)` ×2, aliases, descriptions, a version tag, the versions table, load by alias, UI steps |
| P4 | `mlflow models serve` command; `scoring_server.init(model)` + `TestClient`; BK218304 and five realistic mistakes; `app.py`'s three pieces defined in-notebook; the side-by-side ✅/❌ table; terminal commands + notebook→`app.py` map |
| P5 | the gate (+0.002 test ROC-AUC, < 50 MB), what changes for next week's arrivals (77 flips), promote, restart, rollback as a Try-it, who moves the alias |
| P6 (P7 in v2) | tracking vs registry vs serving, the workflow in ten calls, recommendations, what comes next |

### Decisions, and the measurements behind them

- **The challenger comes from `train.py`, not the notebook,** so both versions are trained, scored and
  logged by the same code. The hand-off is `--config runs:/<tuning_run_id>/best_config.json` (typed JSON).
- **`pyfunc_predict_fn="predict_proba"` with a signature built from `predict_proba`**, so pyfunc and both
  endpoints return probabilities.
- **Register `models:/m-…`, not `runs:/…/model`.** The runs URI works but warns that MLflow 3 swapped in
  the logged model.
- **Aliases come from `get_registered_model(...).aliases`:** `search_model_versions` returns versions
  with `aliases` empty in 3.16.1. `ModelVersion.model_id` is also `None`; `source` holds `models:/m-…`.
- **Serving in-process.** `mlflow models serve` runs uvicorn on `mlflow.pyfunc.scoring_server.app:app`,
  which is `scoring_server.init(load_model(...))`. The notebook builds the same app and calls it with
  `TestClient`: no port, and no server started by the notebook. The terminal commands are fenced blocks.
- **The five bad requests were chosen by measurement** over next week's 1,090 arrivals with the
  HistGradientBoosting champion:
  - "Online Travel Agent" for `Online TA` flipped 218 decisions;
  - `total_of_special_requests = -1` flipped 292;
  - `lead_time` negated flipped 265;
  - lower-case `customer_type` flipped 224;
  - a `deposit_type` typo flipped none (only 10 non-refundable bookings, all extreme), so it isn't used.

  With XGBoost, the segment-name mistake flips 215 of the 642 `Online TA` bookings from reconfirm to
  not; BK218304 is one of them.
- **`app.py` resolves the alias once, then loads that version number**, so the reported
  `model_version` always matches the answer.
- **`MIN_GAIN = 0.002`:** a single ROC-AUC on 23,660 test rows has a standard error near 0.002
  (Hanley–McNeil), and the paired difference is smaller. The XGBoost challenger's +0.0152 clears it.
- **P5 scores next week's arrivals with both versions before promoting:** 268 → 271 reconfirm calls,
  77 flips (37 off, 40 on). BK218304 moves 0.7255 → 0.7908.

### Verified against MLflow 3.16.1, not written from memory

- **The UI, rendered headless.** Playwright was routed through MLflow's FastAPI app with `TestClient`,
  with no port open.
  - Model training mode: the sidebar is **Runs · Models · Traces · Model registry**. The Runs table's
    **Models** column shows `booking-cancellation v1/v2`.
  - Model page: the **Versions** table (**Version · Registered at · Created by · Tags · Aliases ·
    Description**), with a pencil in each alias cell.
  - Version page:
    - **Source Run**;
    - **Aliases**, with a pencil;
    - **Stage (deprecated): None**;
    - **Schema: Inputs (20), Outputs (1)**;
    - a **Promote model** button. It copies the version to *another registered model* ("promotion
      across environments"); it doesn't move an alias.
  - Alias dialog: **Add/Edit alias for model version N** → **Save aliases**. Adding an alias used
    elsewhere warns "The "X" alias is also being used on version N. Adding it to this version will
    remove it from version N." and moves it.
- **Scoring server:**
  - `/ping` and `/health` return 200.
  - JSON ints are converted by the signature; an int64 DataFrame passed to pyfunc is refused.
  - Text in a number column → 400 `BAD_REQUEST`.
  - A missing column → 400 `INVALID_PARAMETER_VALUE` "Model is missing inputs […]".
- **Stages:** `transition_model_version_stage` and friends carry `@deprecated(since="2.9.0")`.
- **MLflow Projects** is not deprecated in 3.16.1; only the kubernetes backend is marked experimental.

### Deliberately left out

- MLflow Projects (one sentence in v1; v2's P6 covers it), the `MLmodel` file's internals, autolog (lec 5), skops trust lists, stages beyond one line, `mlflow models build-docker`
  (named as next), `mlflow.models.evaluate`, registry webhooks, and model-signature `params`.

### How it was built and executed

- dev3.12 is unchanged: every pinned version was already installed, and the install cell is a no-op
  there.
- Executed with nbclient in dev3.12, the environment on PATH, and `MLFLOW_DISABLE_AGENT_HINT=1`, from
  an empty store: 60 s, 0 errors. The outputs are clean; the only "error" text is P4's 400 message.
- The store is left in the folder so the UI matches the outputs: `mlflow.db` (1 MB) and `mlruns/`
  (6 MB). Final state: `@champion` → version 2, no `@challenger`.
- The mermaid pictures were rendered through mermaid.ink in light and dark themes.

### Caveats

- Run IDs, model IDs and timestamps belong to this execution. Re-running without deleting the store
  registers versions 3 and 4 and moves the aliases to them, while the prose says "version 1 /
  version 2". Delete `mlflow.db` and `mlruns/` before a live run.
- `mlflow.db` and `mlruns/` aren't git-ignored; the folder is untracked.
- The MLflow UI shows an **MLflow Assistant (Beta)** panel on the right; nothing here uses it.

### Pacing of v1 (kept here, not in the notebook)

71 cells, against lec 5's 107 for the same slot: short of 2.5 hours, which is why the Airflow variant
and v2 exist. About 2 hours at a steady pace with discussion:

| Parts | Time |
|---|---|
| P0 – P1 | ~15 min |
| P2 (the search runs ~30 s; open the UI while it runs) | ~25 min |
| P3 | ~25 min |
| P4 | ~35 min |
| P5 | ~15 min |
| P6 | ~5 min |
