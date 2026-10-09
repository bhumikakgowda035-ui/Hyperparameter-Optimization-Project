"""Bayesian-style hyperparameter optimization using Optuna's TPE sampler."""
import time
import pandas as pd
import optuna
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay

from model_utils import RESULTS, load_data, build_model, evaluate_cv, timed_fit, save_json

N_TRIALS = 20
RANDOM_STATE = 42

def objective(trial):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 50, 250, step=50),
        "max_depth": trial.suggest_categorical("max_depth", [None, 5, 8, 12, 16, 20]),
        "min_samples_split": trial.suggest_int("min_samples_split", 2, 10, step=2),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 4),
        "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", 0.5, 0.8]),
        "class_weight": trial.suggest_categorical("class_weight", [None, "balanced"]),
    }
    score = evaluate_cv(params)
    trial.set_user_attr("cv_accuracy", score)
    return score

def main():
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    X_train, X_test, y_train, y_test = load_data()
    sampler = optuna.samplers.TPESampler(seed=RANDOM_STATE)
    study = optuna.create_study(direction="maximize", sampler=sampler, study_name="random_forest_tuning")
    start = time.perf_counter()
    study.optimize(objective, n_trials=N_TRIALS)
    search_seconds = time.perf_counter() - start

    rows = []
    best_so_far = -1
    for t in study.trials:
        score = float(t.value)
        best_so_far = max(best_so_far, score)
        rows.append({"trial": t.number + 1, "cv_accuracy": score,
                     "best_cv_accuracy_so_far": best_so_far,
                     "elapsed_seconds": t.duration.total_seconds() if t.duration else None,
                     **t.params})
    pd.DataFrame(rows).to_csv(RESULTS / "bayesian_trials.csv", index=False)

    model = build_model(study.best_params)
    fit_seconds = timed_fit(model, X_train, y_train)
    predictions = model.predict(X_test)
    test_acc = accuracy_score(y_test, predictions)
    save_json(RESULTS / "bayesian_best.json", {
        "method": "Bayesian Optimization (Optuna TPE)",
        "best_cv_accuracy": float(study.best_value), "test_accuracy": float(test_acc),
        "search_seconds": search_seconds, "final_fit_seconds": fit_seconds,
        "best_params": study.best_params
    })
    (RESULTS / "bayesian_classification_report.txt").write_text(
        classification_report(y_test, predictions, target_names=["malignant", "benign"]),
        encoding="utf-8"
    )
    disp = ConfusionMatrixDisplay(confusion_matrix(y_test, predictions),
                                  display_labels=["malignant", "benign"])
    disp.plot()
    plt.title("Bayesian Optimization - Confusion Matrix")
    plt.tight_layout()
    plt.savefig(RESULTS / "bayesian_confusion_matrix.png", dpi=150)
    plt.close()

    plt.figure()
    plt.plot([r["trial"] for r in rows], [r["best_cv_accuracy_so_far"] for r in rows], marker="o")
    plt.xlabel("Bayesian Optimization trial")
    plt.ylabel("Best CV accuracy so far")
    plt.title("Bayesian Optimization Convergence")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(RESULTS / "bayesian_convergence.png", dpi=150)
    plt.close()
    print(f"Bayesian best CV={study.best_value:.4f}; test={test_acc:.4f}; search={search_seconds:.2f}s")
    print("Saved outputs in results/")

if __name__ == "__main__":
    main()
