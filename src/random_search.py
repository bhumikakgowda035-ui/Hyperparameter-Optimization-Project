"""Baseline evaluation and Random Search hyperparameter tuning."""
import json
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import ParameterSampler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay

from model_utils import RESULTS, load_data, build_model, evaluate_cv, timed_fit, save_json

N_ITER = 20
RANDOM_STATE = 42

def main():
    X_train, X_test, y_train, y_test = load_data()
    print("Dataset loaded. Training rows:", len(X_train), "| Test rows:", len(X_test))

    # Baseline model
    baseline = build_model()
    baseline_time = timed_fit(baseline, X_train, y_train)
    baseline_acc = accuracy_score(y_test, baseline.predict(X_test))
    baseline_cv = evaluate_cv()
    baseline_result = {
        "method": "Baseline", "best_cv_accuracy": baseline_cv,
        "test_accuracy": float(baseline_acc), "fit_seconds": float(baseline_time),
        "best_params": baseline.get_params()
    }
    save_json(RESULTS / "baseline.json", baseline_result)
    print(f"Baseline: CV={baseline_cv:.4f}, test={baseline_acc:.4f}, fit={baseline_time:.2f}s")

    search_space = {
        "n_estimators": [50, 100, 150, 200, 250],
        "max_depth": [None, 5, 8, 12, 16, 20],
        "min_samples_split": [2, 4, 6, 8, 10],
        "min_samples_leaf": [1, 2, 3, 4],
        "max_features": ["sqrt", "log2", 0.5, 0.8],
        "class_weight": [None, "balanced"],
    }
    candidates = list(ParameterSampler(search_space, n_iter=N_ITER, random_state=RANDOM_STATE))
    rows, best_score, best_params = [], -1, None
    start_all = time.perf_counter()

    for i, params in enumerate(candidates, start=1):
        start = time.perf_counter()
        score = evaluate_cv(params)
        elapsed = time.perf_counter() - start
        rows.append({"trial": i, "cv_accuracy": score, "elapsed_seconds": elapsed, **params})
        if score > best_score:
            best_score, best_params = score, params
        print(f"Random Search {i}/{N_ITER}: CV accuracy={score:.4f}")

    total_seconds = time.perf_counter() - start_all
    model = build_model(best_params)
    fit_seconds = timed_fit(model, X_train, y_train)
    predictions = model.predict(X_test)
    test_acc = accuracy_score(y_test, predictions)
    pd.DataFrame(rows).to_csv(RESULTS / "random_search_trials.csv", index=False)
    save_json(RESULTS / "random_search_best.json", {
        "method": "Random Search", "best_cv_accuracy": best_score,
        "test_accuracy": float(test_acc), "search_seconds": total_seconds,
        "final_fit_seconds": fit_seconds, "best_params": best_params
    })
    (RESULTS / "random_search_classification_report.txt").write_text(
        classification_report(y_test, predictions, target_names=["malignant", "benign"]),
        encoding="utf-8"
    )
    disp = ConfusionMatrixDisplay(confusion_matrix(y_test, predictions),
                                  display_labels=["malignant", "benign"])
    disp.plot()
    plt.title("Random Search - Confusion Matrix")
    plt.tight_layout()
    plt.savefig(RESULTS / "random_search_confusion_matrix.png", dpi=150)
    plt.close()

    plt.figure()
    plt.plot([r["trial"] for r in rows], np.maximum.accumulate([r["cv_accuracy"] for r in rows]),
             marker="o")
    plt.xlabel("Random Search trial")
    plt.ylabel("Best CV accuracy so far")
    plt.title("Random Search Convergence")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(RESULTS / "random_search_convergence.png", dpi=150)
    plt.close()
    print(f"Random Search best CV={best_score:.4f}; test={test_acc:.4f}; search={total_seconds:.2f}s")
    print("Saved outputs in results/")

if __name__ == "__main__":
    main()
