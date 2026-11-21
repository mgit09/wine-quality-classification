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
| Scatter plots | `scatter_<feature>_<feature>.png` (only one pair so far) | partial |
| Correlation matrix | `correlation_matrix.png` | done |
| Interpretation of every figure | Report text | todo |
| Missing values | `s3_preprocessing.py` validates there are none | done |
| Outliers: detection and decision | Boxplots exist, but no decision is recorded (keep, cap or remove, and why) | partial |
| Class imbalance | `class_distribution.png` | done |
| Categorical encoding | Not needed, all features are numeric: say so explicitly | todo (write) |
| Scaling | Done inside the model pipelines, to avoid leakage: explain | todo (write) |
| Duplicate removal and train/test split | `s3_preprocessing.py`: duplicates removed before a stratified 80/20 split. Report how many rows were removed | done |

### 3.3 Discriminant analysis, resampling and PCA

| Requirement | Support in the repo | Status |
|---|---|---|
| Discriminant analysis: relevance, class separation, assumptions | `s4_discriminant_analysis.py`: skewness and standard deviation by class, correlated pairs, LDA vs QDA, `lda_projection.png` | done |
| Resampling: cross-validation or bootstrap, justified | `evaluation.py` (stratified 5-fold CV on the training set) | partial: write the justification |
| Distinguish resampling from class balancing | Explain: CV estimates performance, balancing changes the training weights or priors (the docstring of `s6_class_balancing.py` has the argument) | todo (write) |
| Class balancing, if needed | `s6_class_balancing.py`: class weights (logistic regression, random forest) and equal priors (LDA) compared with the unbalanced version, `class_balancing_comparison.csv`. SMOTE was not tried (needs `imbalanced-learn`), say so in the limitations | done |
| PCA: need, preprocessing, number of components, variance, effect | `s5_pca.py`: standardized first, 8 components reach 90% (Kaiser criterion keeps 3), PCA lowered every model in CV macro F1, so it is not used in the final models | done |

### 3.4 Training and model comparison

| Requirement | Support in the repo | Status |
|---|---|---|
| At least 2 methods, one taught in class, both suited to the task | LDA, logistic regression, random forest in `models.py`, all tuned in `s7_model_tuning.py` | done |
| Reason for each method and its main parameters | Parameter grids in `models.py`, search results in `tuning_<model>.csv` and `tuning_best_parameters.csv` | partial: write the reasons |
| Training and evaluation procedure | Grid search with stratified 5-fold CV on the training set, scored by macro F1 (`s7_model_tuning.py`). One final evaluation on the test set (`s8_final_evaluation.py`) | done |
| Same conditions for all models | Same split, same CV splitter, same scoring, balancing on for all (`s7_model_tuning.py`) | done |
| Accuracy, precision, recall, F1 | `test_comparison.csv` (macro averages and a majority-class baseline) and `test_report_<model>.csv` (per class) | done (not committed yet) |
| Confusion matrix | `confusion_matrix_<model>.png`, one per model | done (not committed yet) |
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

Tuning results (grid search, 5-fold CV on the training set, macro F1):

| Model | Best parameters | CV macro F1 |
|---|---|---|
| LDA (equal priors) | `shrinkage="auto"`, `lsqr` solver | 0.547 |
| Logistic regression (balanced) | `C=100` | 0.544 |
| Random forest (balanced) | no depth limit, `min_samples_leaf=1`, 500 trees | 0.600 |

These scores are optimistic, because the same folds choose the parameters and score them. Only the test set gives the final performance. LDA and logistic regression are almost flat across their grids, so tuning changes little for them. The random forest's best setting is at the edge of its grid.

Test set results (1064 wines, used once):

| Model | Accuracy | Macro precision | Macro recall | Macro F1 | 95% CI of macro F1 |
|---|---|---|---|---|---|
| Majority class | 0.437 | 0.146 | 0.333 | 0.203 | 0.193 to 0.212 |
| LDA | 0.587 | 0.577 | 0.616 | 0.581 | 0.550 to 0.610 |
| Logistic regression | 0.571 | 0.561 | 0.602 | 0.564 | 0.534 to 0.593 |
| Random forest | 0.635 | 0.624 | 0.631 | 0.627 | 0.595 to 0.656 |

Paired bootstrap of the macro F1 difference (`test_paired_differences.csv`): random forest beats LDA by 0.046 (95% interval 0.016 to 0.076) and logistic regression by 0.062 (0.032 to 0.092), so its lead is real even though the separate intervals overlap. LDA vs logistic regression is borderline (0.016, interval 0.000 to 0.034). The random forest is also the best model by cross-validation, which is how the model is chosen without touching the test set.

Per-class F1 on the test set: the random forest handles the medium class much better than LDA (0.597 against 0.492), because LDA with equal priors over-predicts the high class (recall 0.703, precision 0.440). The random forest confuses mostly neighbouring classes (low vs medium, medium vs high) and almost never low vs high (12 wines each way). The most important features of the forest are alcohol, density and volatile acidity.

PCA results (5-fold CV on the training set, macro F1):

| Model | Original features | PCA (8 components) |
|---|---|---|
| LDA (equal priors) | 0.545 | 0.519 |
| Logistic regression (balanced) | 0.544 | 0.523 |
| Random forest (balanced) | 0.597 | 0.567 |

All three models use the balanced settings of the final comparison, so these numbers are comparable with the tuning and test tables. QDA was only compared with LDA in the discriminant analysis step (`s4`), where it scored 0.522 against LDA's 0.554 with class-proportion priors, and was not carried to the final comparison because LDA beat it and the two share the same assumptions.

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
