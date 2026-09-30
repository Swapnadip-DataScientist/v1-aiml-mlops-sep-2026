"""Train Maré Hotels' booking-cancellation model and track the run in MLflow.

    python train.py                                          # the configuration picked in the notebook
    python train.py --max-leaf-nodes 63 --learning-rate 0.2  # any other configuration
    MLFLOW_TRACKING_URI=http://mlflow.internal:5000 python train.py   # log to a shared server

Every run records the config (params + config.json), the data file it read (path + digest),
the git commit (automatic inside a git repo), test metrics, model size, and the fitted
pipeline with its signature. The last lines printed are the run ID and model URI, for
whatever runs next: CI, a registry step, a scoring job.
"""

import argparse
import os
import pickle
import time
import warnings
from pathlib import Path

import mlflow
import pandas as pd
from mlflow.models import infer_signature
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

DATA = Path(__file__).parent / "data" / "bookings_2015_2017.csv.gz"
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


def parse_args():
    ap = argparse.ArgumentParser(description="Train and track the cancellation model.")
    ap.add_argument("--learning-rate", type=float, default=0.1)
    ap.add_argument("--max-leaf-nodes", type=int, default=127)
    ap.add_argument("--max-iter", type=int, default=300)
    ap.add_argument("--experiment", default="booking-cancellations")
    ap.add_argument("--run-name", default="final-hist_gb")
    return ap.parse_args()


def build_pipeline(config):
    prep = ColumnTransformer([
        ("num", make_pipeline(SimpleImputer(strategy="median"), StandardScaler()), NUMERIC),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
    ])
    model = HistGradientBoostingClassifier(
        learning_rate=config["learning_rate"],
        max_leaf_nodes=config["max_leaf_nodes"],
        max_iter=config["max_iter"],
        random_state=SEED,
    )
    return Pipeline([("prep", prep), ("model", model)])


def evaluate(pipe, X, y, prefix):
    proba = pipe.predict_proba(X)[:, 1]
    pred = (proba >= 0.5).astype(int)
    return {
        f"{prefix}_roc_auc": roc_auc_score(y, proba),
        f"{prefix}_pr_auc": average_precision_score(y, proba),
        f"{prefix}_f1": f1_score(y, pred),
        f"{prefix}_precision": precision_score(y, pred),
        f"{prefix}_recall": recall_score(y, pred),
    }


def main():
    args = parse_args()
    config = {
        "model_type": "hist_gb",
        "learning_rate": args.learning_rate,
        "max_leaf_nodes": args.max_leaf_nodes,
        "max_iter": args.max_iter,
    }

    bookings = pd.read_csv(DATA)
    # float64, not int64: the logged signature then says `double`, so a missing value
    # reaches the imputer instead of being rejected by schema enforcement
    X = bookings[FEATURES].astype({c: "float64" for c in NUMERIC})
    y = bookings["is_canceled"]
    X_train, X_rest, y_train, y_rest = train_test_split(
        X, y, test_size=0.4, stratify=y, random_state=SEED)
    X_val, X_test, y_val, y_test = train_test_split(
        X_rest, y_rest, test_size=0.5, stratify=y_rest, random_state=SEED)
    X_trainval, y_trainval = pd.concat([X_train, X_val]), pd.concat([y_train, y_val])

    # the environment decides where runs go; the default is the notebook's local store
    mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db"))
    mlflow.set_experiment(args.experiment)

    with mlflow.start_run(run_name=args.run_name) as run:
        mlflow.log_params(config)
        mlflow.log_dict(config, "config.json")
        mlflow.set_tags({"stage": "pipeline", "entry_point": "train.py"})
        # MLflow 3.16's dataset logging warns about two things that don't apply here: a local
        # path "matches" the same source type twice, and the raw file's integer columns trigger
        # a hint meant for model signatures (this model's inputs are float64 on purpose)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            dataset = mlflow.data.from_pandas(
                bookings, source=str(DATA), name=DATA.name, targets="is_canceled")
            mlflow.log_input(dataset, context="training")

        start = time.time()
        pipe = build_pipeline(config).fit(X_trainval, y_trainval)
        mlflow.log_metric("fit_time_s", time.time() - start)
        mlflow.log_metric("model_size_mb", len(pickle.dumps(pipe)) / 1e6)
        mlflow.log_metrics(evaluate(pipe, X_test, y_test, prefix="test"))

        model_info = mlflow.sklearn.log_model(
            pipe,
            name="model",
            signature=infer_signature(X_trainval, pipe.predict(X_trainval)),
            input_example=X_trainval.head(5),
            serialization_format="cloudpickle",
        )

    print(f"tracking_uri = {mlflow.get_tracking_uri()}")
    print(f"run_id       = {run.info.run_id}")
    print(f"model_uri    = {model_info.model_uri}")


if __name__ == "__main__":
    main()
