# Report and Presentation Requirements

Source: `docs/assignment_instructions.pdf`. Deadline and presentation date: **24/11/2026**. Worth **30 points**, groups of up to 5 students.

The grading looks at **rigor and coherence, not model performance**. Every decision must be justified, every figure must be interpreted in the text, and every technique that is not used must be justified by the data. Code or results alone do not count.

Status legend: **done** (output exists in the repo), **partial**, **todo**.

## 1. Deliverables

| Deliverable | Format | Notes |
|---|---|---|
| Technical report | IEEE article | Folder `reports/` (not created yet) |
| Presentation | PowerPoint | Folder `presentation/` (not created yet). Up to 10 minutes plus 5 minutes of questions |

## 2. Report structure (IEEE)

Required sections, in this order:

1. Title and authors
2. Abstract and keywords
3. Introduction and problem definition
4. Data description and exploratory analysis
5. Methodology
6. Results and discussion
7. Conclusions
8. Limitations
9. References

## 3. Content checklist by assignment section

### 3.1 Data and problem definition

| Requirement | Where in the report | Support in the repo | Status |
|---|---|---|---|
| Dataset name, identifier or version, and link | Introduction | wine_quality, OpenML id 287, https://www.openml.org/d/287 (README) | done |
| Context, number of observations, available variables | Data description | `s1_inspect_data.py` | done |
| Supervised or unsupervised, and why | Introduction | Supervised, decided in the group chat | todo (write) |
| Target variable and task type | Introduction | `quality`, classification into low (<=5), medium (6), high (>=7), `config.py` | done |
| Why the class cut-offs were chosen | Methodology | Scores 5 to 7 hold about 93% of the wines, so the cut follows the distribution of the grades | todo (write) |

### 3.2 Exploratory analysis and preparation

| Requirement | Support in the repo | Status |
|---|---|---|
| Descriptive statistics | `results/tables/descriptive_statistics.csv`. The EDA runs on the deduplicated data (5318 wines, 1179 duplicates removed), the same wines the models use | done |
| Histograms | `feature_histograms.png` | done |
| Scatter plots | `scatter_<feature>_<feature>.png`, colored by class: alcohol vs density (the two features most correlated with quality) and the three most correlated feature pairs (free vs total SO₂ 0.72, density vs alcohol 0.67, residual sugar vs density 0.52), which show redundancy between features | done |
| Features by class | `feature_boxplots_by_class.png`: alcohol rises from low to high quality, volatile acidity, chlorides and density fall, while pH, sulphates, citric acid and residual sugar barely change. The largest isolated points (density 1.039, residual sugar 65.8) belong to one medium wine | done |
| Scatter matrix | `scatter_matrix.png`: 6x6 matrix of the 6 features most correlated with quality (alcohol, density, volatile acidity, chlorides, citric acid, fixed acidity), coloured by class, with per-class histograms on the diagonal. Alcohol separates the classes best (high wines have high alcohol and low density). The classes overlap strongly in every pair and the medium class sits between the others, so no pair separates the three classes cleanly | done |
| Redundancy check | `feature_correlation_pairs.csv`: absolute correlation of the 55 feature pairs, flagged above 0.8 (the redundancy threshold used in the course exercise). No pair exceeds it, so no feature is removed. The largest are free vs total SO₂ (0.72), density vs alcohol (0.67) and residual sugar vs density (0.52) | done |
| Correlation matrix | `correlation_matrix.png` | done |
| Interpretation of every figure | Report text | todo |
| Missing values | `s3_preprocessing.py` validates there are none | done |
| Outliers: detection and decision | `feature_boxplots.png`, and `s9_preprocessing_experiments.py` (`preprocessing_experiments.csv`). Decision: **keep every wine**. 54 of 4254 training rows lie beyond 5 standard deviations, and 21.9% of the training rows have at least one IQR outlier, so removal is not viable. Removing the extreme rows changed the CV macro F1 by -0.004 to -0.001 for LDA, logistic regression and random forest, and only helped QDA (+0.021, fold std 0.014), which is not in the final comparison. The extreme points are real measurements that the models must handle, and nothing proves they are errors | done |
| Skewed features: detection and decision | Skewness by class in `skewness_by_class.csv`. Six features have skewness above 1 (chlorides 5.5, residual sugar 1.8, sulphates 1.8, fixed acidity 1.7, free SO₂ 1.5, volatile acidity 1.5). Decision: **no transform**. A log transform changed the CV macro F1 by +0.006 (LDA), +0.005 (logistic regression), 0.000 (QDA) and -0.001 (random forest), and a Yeo-Johnson power transform (Box-Cox is not usable because citric acid has zeros) by +0.002 (LDA), +0.002 (logistic regression), +0.028 (QDA) and -0.001 (random forest). All changes are within the fold std (0.005 to 0.018) except the power transform for QDA, a single model, so the rule (most models) is not met. LDA assumptions stay approximately met, and the report must say so | done |
| Class imbalance | `class_distribution.png` | done |
| Categorical encoding | Not needed, all features are numeric: say so explicitly | todo (write) |
| Scaling | Done inside the model pipelines, to avoid leakage: explain | todo (write) |
| Duplicate removal and train/test split | `s3_preprocessing.py`: duplicates removed before a stratified 80/20 split. Report how many rows were removed | done |

### 3.3 Discriminant analysis, resampling and PCA

| Requirement | Support in the repo | Status |
|---|---|---|
| Discriminant analysis: relevance, class separation, assumptions | `s4_discriminant_analysis.py`: skewness and standard deviation by class, correlated pairs, LDA vs QDA, `lda_projection.png` | done |
| Resampling: cross-validation or bootstrap, justified | `evaluation.py` (stratified 5-fold CV on the training set, bootstrap for the test intervals) and `s10_resampling_comparison.py` (`resampling_comparison.csv`), which compares hold-out, 5-fold, 10-fold and leave-one-out with the numbers below | done (write the text) |
| Distinguish resampling from class balancing | Explain: CV estimates performance, balancing changes the training weights or priors (the docstring of `s6_class_balancing.py` has the argument) | todo (write) |
| Class balancing, if needed | `s6_class_balancing.py`: class weights (logistic regression, random forest) and equal priors (LDA) compared with the unbalanced version, `class_balancing_comparison.csv`. Random over- and undersampling inside each training fold were also compared (see the resampling paragraph below). SMOTE was not tried (needs `imbalanced-learn`), say so in the limitations | done |
| PCA: need, preprocessing, number of components, variance, effect | `s5_pca.py`: standardized first, 8 components reach 90% (Kaiser criterion keeps 3), PCA lowered every model in CV macro F1, so it is not used in the final models. It also saved no time: the 5-fold CV took 0.04 to 0.05 s (LDA), 0.07 to 0.08 s (logistic regression) and 0.55 to 0.56 s (random forest) with or without PCA (`pca_fit_time.csv`, wall time, machine-dependent), because the dataset is small | done |

### 3.4 Training and model comparison

| Requirement | Support in the repo | Status |
|---|---|---|
| At least 2 methods, one taught in class, both suited to the task | LDA, logistic regression, random forest in `models.py`, all tuned in `s7_model_tuning.py` | done |
| Reason for each method and its main parameters | Parameter grids in `models.py`, search results in `tuning_<model>.csv` and `tuning_best_parameters.csv` | partial: write the reasons |
| Training and evaluation procedure | Grid search with stratified 5-fold CV on the training set, scored by macro F1 (`s7_model_tuning.py`). One final evaluation on the test set (`s8_final_evaluation.py`) | done |
| Same conditions for all models | Same split, same CV splitter, same scoring, balancing on for all (`s7_model_tuning.py`) | done |
| Accuracy, precision, recall, F1 | `test_comparison.csv` (macro averages and a majority-class baseline) and `test_report_<model>.csv` (per class) | done |
| ROC curve and AUC (taught in Aula 11) | `roc_curves.png` (one-vs-rest, one panel per class) and the `auc_*` columns of `test_comparison.csv` | done |
| Confusion matrix | `confusion_matrix_<model>.png`, one per model | done |
| Discussion of differences, limitations and fit to the goal | Report text | todo |
| Best model chosen from the metrics and the context | Report text | todo |

Balancing results (5-fold CV on the training set, out-of-fold predictions):

| Model | Balancing | Accuracy | Macro F1 | Recall high | Recall medium |
|---|---|---|---|---|---|
| LDA | class proportions as priors | 0.576 | 0.554 | 0.388 | 0.613 |
| LDA | equal priors | 0.548 | 0.545 | 0.706 | 0.362 |
| Logistic regression | no balancing | 0.576 | 0.549 | 0.352 | 0.621 |
| Logistic regression | balanced weights | 0.549 | 0.544 | 0.713 | 0.348 |
| Random forest | no balancing | 0.604 | 0.581 | 0.402 | 0.634 |
| Random forest | balanced weights | 0.601 | 0.597 | 0.602 | 0.519 |

Balancing doubles the recall of the high class for LDA and logistic regression, at the cost of the medium class and about 3 points of accuracy. The report must state this trade-off.

Balancing by resampling (Aula 17), same 5-fold CV, resampling applied only to the training part of each fold (validation folds keep the real proportions and contain no copies), models without class weights or priors; the first row of each pair repeats the weight/prior variant for comparison:

| Model | Balancing | Accuracy | Macro F1 | Recall high | Recall medium |
|---|---|---|---|---|---|
| LDA | equal priors | 0.548 | 0.545 | 0.706 | 0.362 |
| LDA | random oversampling | 0.549 | 0.546 | 0.715 | 0.363 |
| LDA | random undersampling | 0.547 | 0.544 | 0.716 | 0.365 |
| Logistic regression | balanced weights | 0.549 | 0.544 | 0.713 | 0.348 |
| Logistic regression | random oversampling | 0.546 | 0.541 | 0.718 | 0.339 |
| Logistic regression | random undersampling | 0.546 | 0.541 | 0.721 | 0.341 |
| Random forest | balanced weights | 0.601 | 0.597 | 0.602 | 0.519 |
| Random forest | random oversampling | 0.588 | 0.583 | 0.602 | 0.494 |
| Random forest | random undersampling | 0.556 | 0.553 | 0.726 | 0.363 |

For LDA and logistic regression, resampling is equivalent to equal priors or class weights (differences up to 0.003, inside the fold std of 0.005 to 0.008): reweighting the loss and duplicating rows produce practically the same classifier. For the random forest, balanced weights is best, oversampling is slightly lower (0.014, about the size of the fold std 0.018, since duplicated rows let the trees memorize) and undersampling is clearly worse (about 0.04) because it keeps only the smallest class size for every class (about 57% of the rows) and throws the rest away. The medium class always pays, and the high class gains. SMOTE (synthetic neighbours) was not run because `imbalanced-learn` is not installed and not in the pinned requirements, so only random resampling was evaluated.

Tuning results (grid search, 5-fold CV on the training set, macro F1):

| Model | Best parameters | CV macro F1 |
|---|---|---|
| LDA (equal priors) | `shrinkage="auto"`, `lsqr` solver | 0.547 |
| Logistic regression (balanced) | `C=100` | 0.544 |
| Random forest (balanced) | no depth limit, `min_samples_leaf=1`, 500 trees, `max_features="sqrt"` | 0.600 |

These scores are optimistic, because the same folds choose the parameters and score them. Only the test set gives the final performance. LDA and logistic regression are almost flat across their grids, so tuning changes little for them. The random forest grid also tuned `max_features` (sqrt, 0.5, 0.75 and 1.0, that is 3, 5, 8 and 11 of the 11 features; log2 was left out because it equals sqrt here). The best setting stayed the default `sqrt` with the same CV macro F1 of 0.600, and the other values were within one fold std (about 0.017), so `max_features` barely matters. The best setting is still at the edge of the grid (no depth limit, `min_samples_leaf=1`, 500 trees). The grid was revised once and not widened further: the edge suggests a larger forest might gain a little, inside the CV noise.

Test set results (1064 wines, used once):

| Model | Accuracy | Macro precision | Macro recall | Macro F1 | 95% CI of macro F1 | Macro AUC |
|---|---|---|---|---|---|---|
| Majority class | 0.437 | 0.146 | 0.333 | 0.203 | 0.193 to 0.212 | 0.500 |
| LDA | 0.587 | 0.577 | 0.616 | 0.581 | 0.550 to 0.610 | 0.775 |
| Logistic regression | 0.571 | 0.561 | 0.602 | 0.564 | 0.534 to 0.593 | 0.771 |
| Random forest | 0.635 | 0.624 | 0.631 | 0.627 | 0.595 to 0.656 | 0.808 |

AUC is one-vs-rest (each class against the other two, using the predicted probability of that class), averaged over the classes. Per class, the random forest scores 0.861 (high), 0.856 (low) and 0.707 (medium), LDA 0.834, 0.828 and 0.663, logistic regression 0.830, 0.828 and 0.655. The ranking of the models is the same as with the F1, and the medium class is the hardest for all of them: it sits between the other two, so no threshold separates it well. The ROC curves show the whole trade-off between true and false positives, while the confusion matrices show one operating point, the one chosen by the class weights or equal priors.

Paired bootstrap of the macro F1 difference (`test_paired_differences.csv`): random forest beats LDA by 0.046 (95% interval 0.016 to 0.076) and logistic regression by 0.062 (0.032 to 0.092), so its lead is real even though the separate intervals overlap. LDA vs logistic regression is borderline (0.016, interval 0.000 to 0.034). The random forest is also the best model by cross-validation, which is how the model is chosen without touching the test set.

Per-class F1 on the test set: the random forest handles the medium class much better than LDA (0.597 against 0.492), because LDA with equal priors over-predicts the high class (recall 0.703, precision 0.440). The random forest confuses mostly neighbouring classes (low vs medium, medium vs high) and almost never low vs high (12 wines each way). Impurity importance (MDI) ranks alcohol, density and volatile acidity highest, but permutation importance on the test set (drop in macro F1, 10 repeats; `random_forest_permutation_importances.csv`, `random_forest_importance_comparison.png`) ranks alcohol (0.152), volatile acidity (0.046) and free SO₂ (0.025) highest. The two agree on alcohol and volatile acidity but not on density, which MDI ranks second and permutation seventh (0.013). MDI favours continuous features with many distinct values and density is correlated with alcohol, so the permutation ranking is the more reliable one. Residual sugar, fixed acidity and citric acid add almost nothing. The permutation importance is computed on the test set only for interpretation, after every decision was made.

PCA results (5-fold CV on the training set, macro F1):

| Model | Original features | PCA (8 components) |
|---|---|---|
| LDA (equal priors) | 0.545 | 0.519 |
| Logistic regression (balanced) | 0.544 | 0.523 |
| Random forest (balanced) | 0.597 | 0.567 |

All three models use the balanced settings of the final comparison, so these numbers are comparable with the tuning and test tables. QDA was only compared with LDA in the discriminant analysis step (`s4`), where it scored 0.522 against LDA's 0.554 with class-proportion priors, and was not carried to the final comparison because LDA beat it and the two share the same assumptions.

Validation methods (training set, macro F1, 10 repetitions with new shuffles; `resampling_comparison.csv`):

| Model | Method | Macro F1 | Std between repeats | Gap to leave-one-out | Fits per run | Seconds per run |
|---|---|---|---|---|---|---|
| LDA | hold-out 80/20 | 0.542 | 0.0086 | -0.006 | 1 | 0.01 |
| LDA | 5-fold | 0.546 | 0.0022 | -0.001 | 5 | 0.02 |
| LDA | 10-fold | 0.546 | 0.0016 | -0.001 | 10 | 0.04 |
| LDA | leave-one-out | 0.547 | not applicable | 0 | 4254 | 15.3 |
| Logistic regression | hold-out 80/20 | 0.536 | 0.0144 | -0.004 | 1 | 0.01 |
| Logistic regression | 5-fold | 0.542 | 0.0016 | +0.002 | 5 | 0.06 |
| Logistic regression | 10-fold | 0.541 | 0.0017 | +0.001 | 10 | 0.11 |
| Logistic regression | leave-one-out | 0.540 | not applicable | 0 | 4254 | 46.3 |
| Random forest | hold-out 80/20 | 0.590 | 0.0163 | not run | 1 | 0.11 |
| Random forest | 5-fold | 0.589 | 0.0067 | not run | 5 | 0.52 |
| Random forest | 10-fold | 0.593 | 0.0062 | not run | 10 | 1.10 |

Reading: a single hold-out split moves the estimate 2 to 9 times more than 5-fold when the shuffle changes (std 0.009 to 0.016), so one split is not reliable. 5-fold and 10-fold give the same mean and almost the same stability, and 10-fold costs twice as much, so 5 folds are enough. Leave-one-out is the least biased reference, but it is 700 to 800 times slower than 5-fold for LDA and logistic regression and agrees with 5-fold within 0.002, so it adds nothing. It was not run for the random forest, which would need 4254 forest fits. The std measures only the change caused by the shuffle, not the sampling variability of the wines (the bootstrap intervals in the test table cover that). The `seconds_per_run` column depends on the machine and is the only column that changes between runs.

Preprocessing experiments (5-fold CV on the training set, macro F1 change against the untreated data, treatments learned inside each fold and every variant scored on the same wines):

| Model | Remove extremes (z > 5) | Log of skewed features | Both | Power transform (Yeo-Johnson) | Fold std (untreated) |
|---|---|---|---|---|---|
| LDA (equal priors) | -0.001 | +0.006 | +0.007 | +0.002 | 0.005 |
| QDA | +0.021 | 0.000 | +0.019 | +0.028 | 0.013 |
| Logistic regression (balanced) | -0.002 | +0.005 | +0.003 | +0.002 | 0.006 |
| Random forest (balanced) | -0.004 | -0.001 | -0.005 | -0.001 | 0.018 |

Rule fixed before the run: adopt a treatment only if it beats the fold std for most models. None did (the power transform beats it only for QDA, 1 of 4 models), so the data is kept as is. The test set was not used.

## 4. Points that must appear in the text

- LDA assumptions (normality inside each class, equal covariance) are only approximately met, and why LDA is still worth applying.
- Why QDA needs standardized inputs in scikit-learn, and why LDA still beat QDA.
- Splitting before any scaling, PCA or resampling, and fitting those inside the CV folds, to avoid leakage.
- The 3-class cut-off is a choice of the group, and it discards the order of the grades.
- Red and white wines are mixed and the dataset has no column telling them apart.
- Tuned CV scores are optimistic. The test set is used once, at the end, for the real estimate.
- LDA needs standardized features to tune a numeric shrinkage: scikit-learn only standardizes for `shrinkage="auto"`. Without a scaler, the numeric values collapsed the model (macro F1 0.36) only because the sulfur dioxide columns dominate the raw covariance.
- Macro averages are used because the high class is only about 19% of the training set.

## 5. Presentation (10 minutes, plus 5 of questions)

Cover only: the problem, the data, the methodology, the main results and the conclusions. Suggested time budget:

| Part | Time |
|---|---|
| Problem and data | 2 min |
| Exploratory analysis and preparation | 2 min |
| Methodology (discriminant analysis, resampling, PCA, models) | 2 min |
| Results and comparison | 3 min |
| Conclusions and limitations | 1 min |

Everyone in the group should be able to answer questions about any part of the work.

## 6. Final check before submitting

- [ ] Dataset name, id and link are in the report
- [ ] Every figure and table has a caption and is interpreted in the text
- [ ] Every technique that was not used has a justification
- [ ] The best model is chosen from the metrics and the context, not only the score
- [ ] The test set was used only once, at the end
- [ ] References include Cortez et al. (2009) and the course material
- [ ] All members are listed
