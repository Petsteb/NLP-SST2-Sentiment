"""
Grid Search for TF-IDF + Logistic Regression on SST-2
=====================================================
Runs all combinations of the parameter grid below,
logs every result to  logreg_grid_results.txt  and prints
a sorted summary at the end.

Usage:
    python grid_search_logreg.py
"""

import json, os, time, itertools, warnings
import numpy as np
import pandas as pd
from datetime import datetime
import random

from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, log_loss,
)
# ──────────────────────────────────────────────
#  PARAMETER GRID  –  edit these lists freely
# ──────────────────────────────────────────────
PARAM_GRID = {
    "C":             [0.01, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0, 2.0, 5.0, 10.0],
    "solver":        ["lbfgs", "liblinear"],
    "ngram_range":   [(1, 1), (1, 2), (1, 3)],
    "max_features":  [50_000, 100_000, 200_000],
    "sublinear_tf":  [True],
}

# Fixed settings
MAX_ITER       = 500        # high enough so the solver always converges
SEED           = 42
DATASET        = "SST-2"
TEXT_COLUMN    = "sentence_preprocessed"
LABEL_COLUMN   = "label"
STRIP_ACCENTS  = "unicode"
ROOT_DIR       = os.path.dirname(os.path.abspath(__file__))
OUTPUT_FILE    = os.path.join(ROOT_DIR, "logreg_grid_results.txt")

# ──────────────────────────────────────────────

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)

def load_data():
    dataset_path = os.path.join(ROOT_DIR, DATASET)
    splits = {}
    for name in ("train", "validation", "test"):
        with open(os.path.join(dataset_path, f"{name}.json"), "r", encoding="utf-8") as f:
            splits[name] = pd.DataFrame(json.load(f))
    return splits["train"], splits["validation"], splits["test"]

def main():
    set_seed(SEED)
    warnings.filterwarnings("ignore")   # suppress convergence warnings during grid search

    # ── Load data ──
    print("Loading data …")
    df_train, df_val, df_test = load_data()
    y_train = df_train[LABEL_COLUMN].values
    y_val   = df_val[LABEL_COLUMN].values

    # ── Build all combos ──
    keys   = list(PARAM_GRID.keys())
    combos = list(itertools.product(*PARAM_GRID.values()))
    total  = len(combos)
    print(f"Total parameter combinations: {total}\n")

    results = []
    header = (
        f"{'#':>3}  {'C':>6}  {'solver':<11}  {'ngram':>7}  {'max_feat':>9}  "
        f"{'sub_tf':>6}  {'train_acc':>9}  {'val_acc':>8}  {'val_prec':>8}  "
        f"{'val_rec':>8}  {'val_f1':>7}  {'val_loss':>8}  {'time_s':>7}"
    )
    sep = "─" * len(header)

    with open(OUTPUT_FILE, "w", encoding="utf-8") as log:
        log.write(f"Logistic Regression Grid Search — {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        log.write(f"Dataset: {DATASET}  |  Seed: {SEED}  |  max_iter: {MAX_ITER}\n")
        log.write(f"Total combos: {total}\n\n")
        log.write(header + "\n")
        log.write(sep + "\n")

        print(header)
        print(sep)

        for idx, values in enumerate(combos, 1):
            params = dict(zip(keys, values))

            C            = params["C"]
            solver       = params["solver"]
            ngram_range  = params["ngram_range"]
            max_features = params["max_features"]
            sublinear_tf = params["sublinear_tf"]

            # Vectorise (needs to be re-done when TF-IDF params change)
            tfidf = TfidfVectorizer(
                max_features=max_features,
                ngram_range=ngram_range,
                sublinear_tf=sublinear_tf,
                strip_accents=STRIP_ACCENTS,
            )
            X_train = tfidf.fit_transform(df_train[TEXT_COLUMN].fillna("").values)
            X_val   = tfidf.transform(df_val[TEXT_COLUMN].fillna("").values)

            # Train
            t0 = time.time()
            model = LogisticRegression(
                C=C, max_iter=MAX_ITER, solver=solver, random_state=SEED
            )
            model.fit(X_train, y_train)
            elapsed = time.time() - t0

            # Evaluate
            y_tr_pred = model.predict(X_train)
            y_vl_pred = model.predict(X_val)

            train_acc  = accuracy_score(y_train, y_tr_pred)
            val_acc    = accuracy_score(y_val, y_vl_pred)
            val_prec   = precision_score(y_val, y_vl_pred, average="macro", zero_division=0)
            val_rec    = recall_score(y_val, y_vl_pred, average="macro", zero_division=0)
            val_f1     = f1_score(y_val, y_vl_pred, average="macro", zero_division=0)
            val_ll     = log_loss(y_val, model.predict_proba(X_val))

            ngram_str = f"{ngram_range[0]}-{ngram_range[1]}"
            row = (
                f"{idx:3d}  {C:6.2f}  {solver:<11}  {ngram_str:>7}  {max_features:>9}  "
                f"{'Y' if sublinear_tf else 'N':>6}  {train_acc:9.4f}  {val_acc:8.4f}  "
                f"{val_prec:8.4f}  {val_rec:8.4f}  {val_f1:7.4f}  {val_ll:8.4f}  {elapsed:7.2f}"
            )
            print(row)
            log.write(row + "\n")

            results.append({
                "C": C, "solver": solver,
                "ngram_range": ngram_str,
                "max_features": max_features,
                "sublinear_tf": sublinear_tf,
                "train_acc": round(train_acc, 4),
                "val_acc": round(val_acc, 4),
                "val_precision": round(val_prec, 4),
                "val_recall": round(val_rec, 4),
                "val_f1": round(val_f1, 4),
                "val_log_loss": round(val_ll, 4),
                "time_s": round(elapsed, 2),
            })

        # ── Sorted summary ──
        log.write(f"\n{'=' * len(header)}\n")
        log.write("TOP 10 BY VALIDATION ACCURACY\n")
        log.write(f"{'=' * len(header)}\n")

        sorted_res = sorted(results, key=lambda r: r["val_acc"], reverse=True)
        for rank, r in enumerate(sorted_res[:10], 1):
            line = (
                f"  #{rank}  C={r['C']:<6}  solver={r['solver']:<11}  "
                f"ngram={r['ngram_range']}  max_feat={r['max_features']}  "
                f"val_acc={r['val_acc']:.4f}  val_f1={r['val_f1']:.4f}  "
                f"val_loss={r['val_log_loss']:.4f}"
            )
            log.write(line + "\n")

        log.write(f"\n{'=' * len(header)}\n")
        log.write("TOP 10 BY VALIDATION F1 (MACRO)\n")
        log.write(f"{'=' * len(header)}\n")

        sorted_f1 = sorted(results, key=lambda r: r["val_f1"], reverse=True)
        for rank, r in enumerate(sorted_f1[:10], 1):
            line = (
                f"  #{rank}  C={r['C']:<6}  solver={r['solver']:<11}  "
                f"ngram={r['ngram_range']}  max_feat={r['max_features']}  "
                f"val_acc={r['val_acc']:.4f}  val_f1={r['val_f1']:.4f}  "
                f"val_loss={r['val_log_loss']:.4f}"
            )
            log.write(line + "\n")

    print(f"\n{'=' * 60}")
    print(f"  Done! {total} combinations tested.")
    print(f"  Results saved to: {OUTPUT_FILE}")
    print(f"{'=' * 60}\n")

    # Print top 5 to console
    print("TOP 5 BY VALIDATION ACCURACY:")
    for rank, r in enumerate(sorted_res[:5], 1):
        print(f"  #{rank}  C={r['C']:<6}  solver={r['solver']:<11}  "
              f"ngram={r['ngram_range']}  max_feat={r['max_features']}  "
              f"val_acc={r['val_acc']:.4f}  val_f1={r['val_f1']:.4f}")


if __name__ == "__main__":
    main()
