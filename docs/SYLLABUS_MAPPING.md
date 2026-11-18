# Syllabus-only model policy

This project follows the ML syllabus screenshot supplied by the user on 18 November 2026.

| Implemented method | Syllabus wording | Implementation |
|---|---|---|
| Logistic Regression | Unit 1: Logistic Regression | StandardScaler + sklearn LogisticRegression |
| ID3 decision tree | Unit 1: Basic algorithm (ID3); dealing with continuous values | Original multiway information-gain ID3; quantile bins learned on fitting data only |
| K-nearest neighbours | Unit 1: k-nearest neighbour learning (classification/regression) | StandardScaler + exact distance-weighted KNN |
| Random Forest | Boosting/Bayesian Learning/HMM: Random Forest | sklearn RandomForestClassifier |
| Gradient Boosting | Combining weak learners; Gradient Boost | Standard sklearn GradientBoostingClassifier with shallow trees |
| Gaussian Naive Bayes | Naive Bayes Classifier | Gaussian continuous-feature likelihoods |
| HMM and Viterbi | Hidden Markov Models (HMMs); required by hackathon | Directed-network states, model emissions and Viterbi dynamic programming |
| Bayesian optimization | Hyperparameter tuning methods: Bayesian Optimization | Optuna tree-structured Parzen estimator, training-trip CV objective |

No Extra Trees, histogram-specific boosting, XGBoost, LightGBM, CatBoost, RBF SVM, pretrained model or external model weights are part of the syllabus-only submitted comparison or deployment. An earlier prototype was archived locally when the user supplied the syllabus. Its measurements are superseded, not presented as the current results.

Nearest-neighbour spatial indexing, coordinate projection, shortest-path graph search, metric calculations, FastAPI and visualization are preprocessing/deployment utilities, not additional trained model families.

Only the supplied training partition is used for fitting and group-aware cross-validation. Student testing is development evaluation. Reserved evaluator inputs are not supplied and are not simulated or hardcoded.
