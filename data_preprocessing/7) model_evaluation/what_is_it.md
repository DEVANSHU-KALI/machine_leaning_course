# Module 06 Roadmap
## 1. Cross-Validation Strategies (Validation Scheme Selection)
- Why a simple train/test split fails on small or skewed data.
- KFold vs. StratifiedKFold vs. TimeSeriesSplit vs. GroupKFold.
- Cross-validation pitfalls and how pipelines protect fold boundaries.

## 2. Hyperparameter Optimization Engines
- GridSearchCV: Exhaustive brute-force search over a parameter grid.
- RandomizedSearchCV: Sampling from parameter distributions (computationally efficient).
- HalvingGridSearchCV / Successive Halving: Discarding unpromising trials early.

## 3. Tuning Preprocessing & Estimator Jointly Inside Pipelines
- Using the <step_name>__<param_name> syntax to optimize feature selection and classifier settings simultaneously.