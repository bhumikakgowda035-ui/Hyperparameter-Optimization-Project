# Project Report: Comparing Random Search and Bayesian Optimization

## 1. Title
**Hyperparameter Optimization in Machine Learning: Random Search vs Bayesian Optimization**

## 2. Abstract
This project compares Random Search and Bayesian Optimization for tuning a Random Forest classification model. The built-in Breast Cancer Wisconsin dataset from scikit-learn is used. A baseline model is evaluated, then both search methods test 20 candidate configurations using five-fold cross-validation. The best configuration from each method is evaluated on a held-out test set. Accuracy, search time, convergence, and confusion matrices are recorded.

## 3. Problem Statement
Machine-learning model performance depends on hyperparameters such as the number of trees, tree depth, and minimum samples per split. Selecting these values manually can be difficult. This project investigates whether Bayesian Optimization can find a high-performing configuration more efficiently than Random Search under the same trial budget.

## 4. Objectives
- Build and evaluate a baseline Random Forest model.
- Implement Random Search for hyperparameter tuning.
- Implement Bayesian-style optimization using Optuna's TPE sampler.
- Compare cross-validation accuracy, test accuracy, execution time, and convergence.
- Save reproducible results and visualizations.

## 5. Tools and Technologies
Python, scikit-learn, NumPy, Pandas, Matplotlib, Optuna, VS Code, GitHub.

## 6. Dataset
The Breast Cancer Wisconsin dataset bundled with scikit-learn is used. It contains numerical features calculated from cell-nuclei measurements and a binary target class. The project uses an 80% training split and a 20% test split with stratification and a fixed random seed. This is a demonstration dataset, not a medical diagnosis tool.

## 7. Methodology
1. Load the dataset and split it into training and test data.
2. Train a baseline Random Forest model.
3. Use five-fold cross-validation to score candidate hyperparameters.
4. Random Search samples 20 combinations from the defined parameter space.
5. Optuna's TPE sampler runs 20 trials, using earlier trial results to guide later choices.
6. Fit each method's best configuration on the full training split.
7. Evaluate on the held-out test split and create plots/reports.

## 8. Hyperparameters
- `n_estimators`: number of trees.
- `max_depth`: maximum tree depth.
- `min_samples_split`: minimum samples required to split an internal node.
- `min_samples_leaf`: minimum samples required at a leaf.
- `max_features`: features considered when splitting.
- `class_weight`: optional class balancing.

## 9. Evaluation Metrics
- **Cross-validation accuracy:** average accuracy across five folds.
- **Test accuracy:** accuracy on held-out test data.
- **Search time:** wall-clock time spent evaluating candidate configurations.
- **Convergence:** best cross-validation score achieved so far after each trial.
- **Confusion matrix:** counts of correct and incorrect class predictions.

## 10. Results
Run the scripts and fill this section from `results/comparison.csv`:
- Baseline test accuracy: ______
- Random Search best CV accuracy: ______
- Random Search test accuracy: ______
- Random Search search time: ______ seconds
- Bayesian Optimization best CV accuracy: ______
- Bayesian Optimization test accuracy: ______
- Bayesian Optimization search time: ______ seconds
- Best method for this run and reason: ______

Do not invent results before running the code. Accuracy and runtime may vary by computer and package version.

## 11. Discussion
Compare the convergence plot and the CSV outputs. Random Search explores configurations without learning from earlier trials. TPE-based Bayesian Optimization uses observations from earlier trials to suggest later configurations. Bayesian Optimization may reach a strong score earlier, but it is not guaranteed to win on every dataset or run. With only 20 trials, results are an educational comparison rather than proof that one method is always better.

## 12. Conclusion
This project demonstrates two practical hyperparameter optimization methods and provides a repeatable workflow for evaluating their performance. The final conclusion should be based on the generated results, including both accuracy and computational cost.

## 13. Future Enhancements
- Compare XGBoost or LightGBM if installed and supported.
- Try additional datasets and metrics such as F1-score.
- Increase the number of optimization trials.
- Export the best model for later predictions.
- Add a simple Flask interface.

## 14. How to Reproduce
Follow the installation and execution instructions in `README.md`. Run `random_search.py`, `bayesian_opt.py`, and then `compare_results.py`.
