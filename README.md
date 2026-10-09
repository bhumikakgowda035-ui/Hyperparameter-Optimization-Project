# Hyperparameter Optimization: Random Search vs Bayesian Optimization

Beginner-friendly BCA project comparing **Random Search** and **Bayesian Optimization (Optuna)** for tuning a Random Forest classifier.

## What this project does
1. Loads the built-in Breast Cancer Wisconsin dataset from scikit-learn (no dataset download needed).
2. Splits data into training and test sets.
3. Trains a baseline Random Forest and records accuracy and execution time.
4. Tunes hyperparameters using Random Search.
5. Tunes hyperparameters using Optuna's Bayesian/TPE sampler.
6. Compares best cross-validation scores, test accuracy, runtime, and convergence plots.
7. Saves results, a confusion matrix, and a classification report in `results/`.

> Note: Optuna's default TPE sampler is a sequential model-based optimization method commonly used for Bayesian optimization. Results can vary slightly between runs.

## 1. Install Python
Install Python 3.10 or newer from https://www.python.org/downloads/ . During installation on Windows, tick **Add Python to PATH**.

## 2. Open the project in VS Code
- Extract this ZIP file.
- Open VS Code.
- Select **File → Open Folder** and choose `hyperparameter_optimization_project`.
- Open **Terminal → New Terminal**.

## 3. Create a virtual environment (recommended)
Windows:
```bash
python -m venv .venv
.venv\Scripts\activate
```
If PowerShell blocks activation, use Command Prompt in VS Code, or run:
```bash
.venv\Scripts\activate.bat
```

## 4. Install required packages
```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## 5. Run the project
From the project root folder:
```bash
python src/random_search.py
python src/bayesian_opt.py
python src/compare_results.py
```
Run them in that order. Each script prints progress and saves files into `results/`.

If `python` is not recognized on Windows, try `py` instead, e.g. `py src/random_search.py`.

## Important files
- `src/model_utils.py` — shared dataset/model helpers.
- `src/random_search.py` — baseline and Random Search.
- `src/bayesian_opt.py` — Optuna optimization.
- `src/compare_results.py` — comparison table and convergence plots.
- `docs/optimization_report.md` — report template and explanation.
- `results/` — output files created after running scripts.

## GitHub upload
1. Create a new repository on GitHub.
2. In VS Code, open the project folder.
3. Run:
```bash
git init
git add .
git commit -m "Add hyperparameter optimization project"
git branch -M main
git remote add origin https://github.com/YOUR-USERNAME/YOUR-REPOSITORY.git
git push -u origin main
```
Replace the URL with your own repository URL. Never share your GitHub password or token.
