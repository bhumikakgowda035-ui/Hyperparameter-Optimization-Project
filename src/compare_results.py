"""Create comparison table and convergence plot after both searches have run."""
import json
import pandas as pd
import matplotlib.pyplot as plt
from model_utils import RESULTS

def read_json(name):
    path = RESULTS / name
    if not path.exists():
        raise FileNotFoundError(f"Missing {path.name}. Run random_search.py and bayesian_opt.py first.")
    return json.loads(path.read_text(encoding="utf-8"))

def main():
    random = read_json("random_search_best.json")
    bayes = read_json("bayesian_best.json")
    baseline = read_json("baseline.json")
    rows = []
    for item in [baseline, random, bayes]:
        rows.append({
            "Method": item["method"],
            "Best CV accuracy": item["best_cv_accuracy"],
            "Test accuracy": item["test_accuracy"],
            "Search time (seconds)": item.get("search_seconds", item.get("fit_seconds", 0)),
            "Final fit time (seconds)": item.get("final_fit_seconds", item.get("fit_seconds", 0)),
            "Best parameters": str(item["best_params"]),
        })
    df = pd.DataFrame(rows)
    df.to_csv(RESULTS / "comparison.csv", index=False)
    df.to_markdown(RESULTS / "comparison_table.md", index=False)
    print("\nCOMPARISON RESULTS\n")
    print(df[["Method", "Best CV accuracy", "Test accuracy", "Search time (seconds)"]].to_string(index=False))

    random_trials = pd.read_csv(RESULTS / "random_search_trials.csv")
    bayes_trials = pd.read_csv(RESULTS / "bayesian_trials.csv")
    plt.figure()
    plt.plot(random_trials["trial"], random_trials["cv_accuracy"].cummax(),
             marker="o", label="Random Search")
    plt.plot(bayes_trials["trial"], bayes_trials["cv_accuracy"].cummax(),
             marker="o", label="Bayesian Optimization (TPE)")
    plt.xlabel("Trial")
    plt.ylabel("Best CV accuracy so far")
    plt.title("Convergence Comparison")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(RESULTS / "convergence_comparison.png", dpi=150)
    plt.close()
    print("\nFiles saved to results/. Open comparison.csv and convergence_comparison.png.")

if __name__ == "__main__":
    main()
