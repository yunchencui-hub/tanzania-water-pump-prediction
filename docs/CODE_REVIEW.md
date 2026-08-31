# Code Review Report

**Project:** Predictive Maintenance for Sustainable Water Infrastructure (Tanzania Water Pump Dataset)
**Code reviewed:** the 8 pipeline scripts in `src/`
**Reference document:** `docs/technical_report.pdf` (DT2 technical report)
**Review date:** 2026-08-31

This review covers every script line by line. It checks three things: whether the code is correct, whether the code matches the claims in the technical report, and whether the code follows the standards of a mature open-source project. Findings are labeled by type: **Bug** (wrong behavior), **Methodology** (a modeling decision that affects the validity of results), **Consistency** (a mismatch between the code and the report, or inside the report), **Portability** (code that only runs on the original machine), **Code quality** (works correctly but should be written better), and **Efficiency** (wastes time or memory).

Section 6 lists the small set of changes that were applied in this repository. Section 7 lists everything that was found but deliberately **not** changed, because changing it would alter the numbers, and the numbers in the report were produced by this exact code.

---

## 1. Summary

The pipeline is methodologically sound in the places that matter most:

1. The train/validation split is stratified with a fixed seed (`random_state=42`) in every script, so both models are evaluated on the same 11,880 validation pumps.
2. The MLP preprocessing avoids data leakage correctly: the split happens first, then `StandardScaler` and `OneHotEncoder` are fitted on the training fold only (`01_nn_data_prep.py`, steps 4–6).
3. Class 2 oversampling is applied only to the training fold, never to the validation fold (`01_nn_data_prep.py`, step 7). This is the correct order and many student projects get it wrong.
4. The social cost arithmetic (5 × FN + 1 × FP for Class 2) is implemented identically in all three places it appears, and I re-computed every social cost value in the report from the published confusion matrices. The XGBoost numbers are exactly right.
5. TR-04 probability validation is a hard gate: `06_confidence_analysis.py` raises an exception if probabilities do not sum to 1, instead of just printing a warning.

The main problems found:

| # | Type | Location | Finding |
|---|------|----------|---------|
| 1 | Methodology | `model_audit.py`, `06_confidence_analysis.py` | XGBoost early stopping monitors the same validation set that is used for the final reported metrics. |
| 2 | Methodology | `01_nn_data_prep.py` → `02_nn_train.py` | Oversampled duplicate rows leak into the MLP's internal early-stopping split. |
| 3 | Consistency | report §VIII.B/C | The MLP confusion matrix in the report does not exactly match the MLP summary metrics in the same report (details in Section 3). |
| 4 | Portability | `data_cleaning.py`, `strategic_eda.py` | Hardcoded absolute paths (`C:\Users\13922\Downloads\...`). Fixed in this repository. |
| 5 | Bug (display only) | `data_cleaning.py` | The summary printed the test set row count minus 1. Fixed in this repository. |
| 6 | Code quality | 3 files | `calculate_social_cost` is copy-pasted into three scripts; the two copies of `validate_probability_output` already differ slightly. |
| 7 | Efficiency | `02_nn_train.py`, `03_nn_tuning.py` | Sparse matrices are converted to dense before MLP training. This is the peak memory cost of the whole pipeline and is not necessary. |
| 8 | Code quality | all files | `warnings.filterwarnings('ignore')` suppresses every warning globally, including ones that would matter. |

None of these change the conclusion of the report. XGBoost with weight 5 still wins on social cost, and the reported XGBoost numbers are verified correct.

---

## 2. What the pipeline does

For a reader who has not read the report: the task is 3-class classification of Tanzanian water pump status (0 = Functional, 1 = Non-Functional, 2 = Needs Repair, only 7.27% of samples). Missing a Class 2 pump is defined to cost 5 times more than a false alarm, so evaluation uses Social Cost = 5 × FN(Class 2) + 1 × FP(Class 2) instead of accuracy.

Execution order and data flow:

```
data_cleaning.py        raw CSVs -> cleaned_train_values.csv, cleaned_train_labels.csv, cleaned_test_values.csv
strategic_eda.py        cleaned CSVs (+ raw CSV for Figs 5-7) -> 6 EDA plots
feature_engineering.py  cleaned CSVs -> final_X_train.parquet, final_y_train.csv, final_X_test.parquet
model_audit.py          parquet -> XGBoost trained at class weights 1/5/10/20 -> audit_results.csv, plots
01_nn_data_prep.py      parquet -> scaled/encoded/oversampled arrays -> nn_preprocessed_data.pkl
03_nn_tuning.py         nn_preprocessed_data.pkl -> 10-combination grid search -> nn_tuning_results.csv
02_nn_train.py          nn_preprocessed_data.pkl -> final MLP (alpha=0.05) -> nn_model.pkl, nn_audit_results.csv
06_confidence_analysis.py  retrains XGBoost W=5, loads MLP -> TR-04 validation, confidence plots
```

---

## 3. Verification of the report numbers against the code

I recomputed the report's key numbers by hand from the confusion matrices printed in the report, and checked every constant in the report against the code.

### 3.1 Constants: all match

| Report claim | Code location | Match |
|---|---|---|
| Social cost = 5 × FN + 1 × FP | `model_audit.py:52-59`, `02_nn_train.py:43-50`, `03_nn_tuning.py:31-40` | Yes |
| Class weights audited: 1, 5, 10, 20 | `model_audit.py:41` (`WEIGHTS_TO_AUDIT`) | Yes |
| TR thresholds 0.50 / 0.80 / 0.10 / 3 columns / 3.0 lift | `model_audit.py:43-48` | Yes |
| XGBoost: 1000 trees, lr 0.05, depth 6, early stopping 50 | `model_audit.py:27-39` (`BASE_PARAMS`) | Yes |
| MLP architecture 128-64-32, ReLU, adam | `02_nn_train.py:27-31`, `03_nn_tuning.py:20` | Yes |
| Tuning grid: alpha {0.0001, 0.001, 0.01, 0.05, 0.1} × lr {0.001, 0.0005} = 10 runs | `03_nn_tuning.py:22-23` | Yes |
| Best alpha 0.05 used for the final MLP | `02_nn_train.py:33` (`ALPHA = 0.05`) | Yes |
| Oversampling factor 5× on Class 2, training fold only | `01_nn_data_prep.py:22, 174-192` | Yes |
| High-cardinality cutoff: drop columns with > 50 unique values | `01_nn_data_prep.py:20` | Yes |
| 80/20 stratified split, seed 42, everywhere | all scripts | Yes |
| Dropped columns: `num_private`, `scheme_name`, `quantity_group`, `wpt_name` | `data_cleaning.py:12-17` | Yes |
| Zeros in `gps_height`/`longitude` set to NaN, no imputation | `data_cleaning.py:19, 122-130` | Yes |
| Zeros kept in `amount_tsh`, `construction_year`, `population` | `data_cleaning.py:21` | Yes |
| `funder`/`installer` reduced to Top-20 + "Other" | `data_cleaning.py:23-24` | Yes |
| Missing construction year replaced by 2000 for `pump_age` only | `feature_engineering.py:18, 71-75` | Yes |
| Hotspot regions Lindi and Mtwara | `feature_engineering.py:20` | Yes |
| Reference year 2014 for pump age | `strategic_eda.py:36`, `feature_engineering.py:16` | Yes |
| Appendix A.2 run order | matches the README run order | Yes |

### 3.2 XGBoost results: arithmetic verified, all correct

From the confusion matrix in report §VIII.B (weight 5×): FN = 248 + 64 = 312, FP = 841 + 351 = 1,192.

- Social cost: 5 × 312 + 1,192 = **2,752**. Matches the report.
- Class 2 recall: 551 / 863 = **0.6385**. Matches.
- Accuracy: (5,228 + 3,139 + 551) / 11,880 = 8,918 / 11,880 = **0.7507**. Matches.
- Lift: 0.6385 / 0.0727 = **8.78**. Matches.
- Weight-sensitivity table (§VI.B): 5×640+102 = 3,302; 5×312+1,192 = 2,752; 5×177+2,226 = 3,111; 5×121+3,360 = 3,965. All four match.

### 3.3 MLP results: the report's confusion matrix and summary table disagree slightly

**Finding (Consistency, affects the PDF only, not the code).** Recomputing from the MLP confusion matrix printed in report §VIII.B (rows 5072/569/811, 930/3318/317, 248/102/513):

| Metric | Computed from the matrix in §VIII.B | Stated in the report | Difference |
|---|---|---|---|
| FP (Class 2) | 811 + 317 = 1,128 | 1,129 (§VIII.C) | off by 1 |
| Social cost | 5 × 350 + 1,128 = 2,878 | 2,879 (abstract, §VII.B, §VIII.C) | off by 1 |
| Accuracy | 8,903 / 11,880 = 0.7494 | 0.7501 (§VII.B, §VIII.C) | off by 0.0007 |
| Class 1 precision | 3,318 / 3,989 = 0.8318 | 0.8351 (§VIII.C) | off by 0.0033 |
| Class 2 caught | 513 | "507 versus 551" in the §VIII.B text | text says 507, matrix says 513 |

The most likely explanation is that the confusion matrix figure and the summary metrics were taken from two different training runs (MLP results shift slightly between runs of different library versions or BLAS builds). The code already contains the fix for this: `02_nn_train.py:230-236` prints a `[REPORT NUMBERS]` block with the matrix and all derived metrics from one single run, guaranteed consistent. If the report is ever revised, all MLP numbers should be copied from that one block. FN = 350, recall = 513/863 = 0.5944, and lift = 8.18 are internally consistent as stated.

One more small mismatch inside the PDF: the abstract states Class 1 precision = 0.8750 for XGBoost, and the table in §VIII.C states 0.8753. Same run, one of the two is a transcription rounding error.

---

## 4. File-by-file review

### 4.1 `src/data_cleaning.py` (Step 1)

**What it does.** Merges training values with labels on `id`, concatenates train and test with an `_is_train` flag, drops 4 useless columns plus `id`, replaces physically impossible zeros in `gps_height`/`longitude` with NaN, keeps informative zeros in `amount_tsh`/`construction_year`/`population`, reduces `installer`/`funder` to Top-20 + "Other", fills missing categoricals with "Unknown", converts `region_code`/`district_code` to strings, maps the target to {0, 1, 2}, splits back into train/test, saves three CSVs.

**Findings.**

1. **Portability (fixed).** The `__main__` block pointed at `C:\Users\13922\Downloads\Training set values.csv` and two similar absolute paths. Nobody else can run this, and it exposes the author's local user directory in a public repository. Replaced with a candidate search over `data/` and the working directory.
2. **Bug, display only (fixed).** The closing summary printed `test_values.shape[0] - 1` as the test row count. Subtracting 1 from the column count is right (it removes the `_is_train` flag added at that point), but subtracting 1 from the row count printed 14,849 instead of 14,850. No computation used this value.
3. **Correct detail worth pointing out.** `reduce_cardinality` accepts a `train_mask` and computes the Top-20 categories from training rows only (`data_cleaning.py:36-54`). This prevents test-set category frequencies from influencing the encoding. This is the right way to do it.
4. **Code quality.** If any target label fails to map, the script prints a warning (`⚠️ WARNING: n unmapped target values!`) and keeps going with NaN labels. This should `raise` instead. It never triggers on this dataset, but a silent NaN target would corrupt everything downstream.
5. **Methodology, minor.** `date_recorded` and `subvillage` are never dropped, converted, or encoded here. Downstream, both are silently discarded in both model branches: `model_audit.py` drops all object columns, and `01_nn_data_prep.py` drops them for exceeding 50 unique values. This is consistent behavior, but it happens implicitly in two different places. The recording date in particular could carry signal (how recently a pump was inspected). Worth an explicit decision and one line of documentation, rather than an accidental drop.
6. **Efficiency, minor.** `reduce_cardinality` uses `df[column].apply(lambda x: x if x in top_categories else other_label)` — a Python-level loop with a list membership test. `df[column].where(df[column].isin(top_categories), other_label)` is vectorized and roughly 20× faster. Runtime here is seconds either way.

### 4.2 `src/strategic_eda.py` (Step 2)

**What it does.** Five analyses on the cleaned data: (1) "sure-kill" hunt — feature values whose Class 1 risk ratio exceeds 2.0; (2) "hidden risk" hunt — features that separate Class 0 from Class 2 in a binary subset; (3) pump age and zero-value risk tests; (4) regional failure hotspots with a 60% threshold; (5) numerical distributions, boxplots, and a correlation heatmap (report Figures 5–7), computed from the **raw** CSV on purpose — the cleaning step converts the invalid zeros to NaN and the region/district codes to strings, which would hide the longitude spike at 0 and the region/district correlation the report describes. The code comment at the top of that section explains this; the reasoning is sound.

**Findings.**

1. **Portability (fixed).** `RAW_TRAIN_VALUES_CANDIDATES` contained the same `C:\Users\13922\Downloads` absolute path. Replaced with `data/` candidates.
2. **Code quality.** `calculate_risk_ratio` loops over every unique feature value and builds a boolean mask per value. A single `groupby` computes the same table in one pass. Correct output, avoidable O(n × k) cost.
3. **Fragile assumption.** `_plot_class_comparison` renames the crosstab columns positionally: `contingency.columns = ['Functional', 'Needs Repair']`. This is only safe because the input is pre-filtered to classes {0, 2}. If someone reuses the function with different classes, the labels will be wrong with no error. Mapping by column value (`contingency.rename(columns={0: 'Functional', 2: 'Needs Repair'})`) would be robust.
4. **Suppressed warnings.** The `groupby(...).apply(lambda ...)` patterns at lines 231 and 303 emit pandas `FutureWarning`s about operating on grouping columns. These are hidden by the global `warnings.filterwarnings('ignore')`. The code still behaves correctly on pandas 2.x, but this pattern is scheduled to change behavior; see cross-cutting finding 5.3.
5. **Verified against the report.** With 10 numeric features, the heatmap annotation condition (`annot=True if len <= 12`) is met, matching Figure 7 which shows annotated cells, and the `region_code`↔`district_code` correlation printed at line 502 is the r = 0.68 the report cites.

### 4.3 `src/feature_engineering.py` (Step 3)

**What it does.** Reloads the cleaned CSVs (forcing `region_code`/`district_code` back to string, since a CSV round-trip loses the dtype), concatenates train and test, creates 16 engineered features (sure-kill flags, hidden-risk flags, zero-value flags, pump age features, hotspot flag, interaction scores), one-hot encodes 24 categorical columns with the encoder fitted on training rows only, drops `recorded_by` and the intermediate `pump_age_bucket`, verifies row alignment with three asserts, and saves Parquet + pickle outputs.

**Findings.**

1. **Efficiency, the biggest one in the pipeline.** The one-hot list includes `lga` (~125 values) and `ward` (~2,092 values). This is what produces the 2,400+ column matrix the report mentions. The encoder is created with `sparse_output=False, dtype=int`, so the result is a **dense** integer matrix of roughly 74,250 × 2,400 ≈ 178 million cells held in memory, then written to Parquet. A sparse matrix would use a small fraction of the memory, and XGBoost consumes sparse input natively. This works on the author's machine and was kept as-is, but it is the first thing to change if the pipeline ever runs on a bigger dataset.
2. **Methodology, disclosed.** `MEDIAN_CONSTRUCTION_YEAR = 2000` is a hardcoded constant, not a computed median. The report states this value openly ("replaced with the median year, 2000"), so this is a documentation-matches-code situation, but if the data ever changes the constant will silently go stale. Computing `int(df.loc[df.construction_year > 0, 'construction_year'].median())` costs one line.
3. **Dead work.** `create_time_features` builds `pump_age_bucket` with `pd.cut`, and `cleanup_features` unconditionally deletes it. The feature never reaches a model. Either use it or stop creating it.
4. **Fragile silent failure.** Every engineered flag depends on exact string values: `quantity == 'dry'`, `payment_type == 'never pay'`, `source_class == 'surface'`. If a future data version renames a category, every flag becomes all-zero with no error. The step already prints the flag counts (for example `is_dry_well: 6,246 pumps`), which is a decent manual check; an assert that each flag is nonzero would make it automatic.
5. **Fragile error handling.** The `except ImportError` fallback at the bottom references `X_train`, which only exists if `engineer_features` already returned. It happens to be safe because the only ImportError that can occur (`pyarrow` missing) is raised at the `to_parquet` call, after `X_train` is assigned, but the handler would crash with a `NameError` if an import failed any earlier. Narrowing the `try` block would remove the trap.
6. **Leftover removed (fixed).** The file ended with a `PHASE 2: XGBOOST PIPELINE` banner comment with no code after it — a leftover from the original combined script. Deleted.
7. **Correct details worth pointing out.** The encoder is fitted with `train_mask` so test-only categories are all-zero instead of leaking; NaN is filled with a `_missing_` sentinel before encoding; and the three asserts at the end catch any row misalignment immediately. All good practice.

### 4.4 `src/model_audit.py` (Step 4 — XGBoost sensitivity analysis and TR audit)

**What it does.** Loads the engineered matrix, drops the leftover object columns, fills NaN with 0, makes the stratified 80/20 split, then trains XGBoost four times with Class 2 sample weights 1/5/10/20. For each weight it computes the confusion matrix, social cost, all five TR checks, and saves a confusion-matrix heatmap. It saves the trade-off dashboard, `audit_results.csv`, and the feature importance table/plot for the weight-5 model.

**Findings.**

1. **Methodology, the most important finding in this review.** `model.fit(..., eval_set=[(X_val, y_val)])` with `early_stopping_rounds=50` means the number of boosting rounds is chosen by watching the same validation set that produces the final reported metrics, and that same set also selects the best weight. The validation set therefore influences the model twice before being reported as unseen data. The reported metrics are mildly optimistic as a result. This is common practice in competition settings and the effect at 11,880 validation samples is small, but a mature setup would use a three-way split: train / early-stopping set / untouched test set. The report does not mention this caveat; it should.
2. **Verified no-leakage claim.** Preprocessing (dropping object columns, `fillna(0)`) happens before the split, and report §IV.F argues this is safe because neither operation uses data statistics. That argument is correct — dropping a column and writing a constant are the same operation regardless of which rows are present.
3. **Compatibility.** `use_label_encoder=False` was removed in XGBoost 2.0; passing it currently triggers a deprecation warning (hidden by the global filter) and may become an error in a future major version. Same for the position of `early_stopping_rounds`, which requires XGBoost ≥ 1.6 in the constructor. This is why `requirements.txt` now pins minimum versions.
4. **Code quality.** `if W == 5: importance_model = model` hardcodes the report's chosen weight. If `WEIGHTS_TO_AUDIT` changes, feature importance silently disappears. Selecting the best-social-cost model after the loop would be robust.
5. **Code quality.** Predictions are computed as `model.classes_[np.argmax(y_val_proba, axis=1)]` here, but with `model.predict()` everywhere else. The two are equivalent; the inconsistency just costs the reader a minute of doubt.
6. **Not saved.** The audited models are thrown away, which forces `06_confidence_analysis.py` to retrain the weight-5 model from scratch (see 4.8.1).

### 4.5 `src/01_nn_data_prep.py` (Step 5 — MLP preprocessing)

**What it does.** Loads the engineered matrix, identifies numeric vs categorical columns, drops categoricals with more than 50 unique values (NFR-02), makes the stratified 80/20 split, fits `StandardScaler` + `OneHotEncoder` in a `ColumnTransformer` on the training fold only, transforms both folds, physically oversamples Class 2 by 5× in the training fold only, shuffles with a seeded RNG, verifies no NaN/Inf, and pickles everything including the fitted preprocessor.

**Findings.**

1. **Correct, and worth stating plainly:** this script does the two things that are easiest to get wrong in this kind of pipeline, and does both right. The split happens before the scaler/encoder are fitted (no statistics leak from validation into training), and the oversampling happens after the split, on the training fold only (no duplicated pump can sit on both sides of the evaluation boundary). Report Figures 9–19 document this sequence and the code matches them exactly.
2. **Methodology, inherited by the next two scripts.** The oversampled duplicates do contaminate one thing: `MLPClassifier` is configured with `early_stopping=True, validation_fraction=0.1`, which carves an internal validation split out of the **oversampled** training data. Copies of the same Class 2 row can land on both sides of that internal split, so the early-stopping score is partly measured on memorized rows and stopping happens later than it should. The final holdout metrics are still honest (the 20% fold never saw oversampling), but training-length decisions and the tuning ranking in `03_nn_tuning.py` are based on an optimistic signal. The fix is to oversample only the 90% that the MLP actually trains on, or to pass a clean validation set. The report's §VIII.D discussion of the TR-03 gap correctly blames oversampling for the inflated training accuracy, but does not mention this internal-split effect.
3. **Code quality.** `identify_column_types` matches on a hardcoded dtype list (`int64/float64/int32/float32`). Any other numeric dtype (for example `int8` from a future optimization) would be misclassified as categorical. `pd.api.types.is_numeric_dtype` covers all of them.
4. **Minor.** The cardinality filter counts unique values on the full dataset before the split. Cardinality is not a target statistic, so there is no real leakage concern; noted only for completeness.

### 4.6 `src/02_nn_train.py` (Step 7 — final MLP training and audit)

**What it does.** Loads the pickled arrays, converts sparse to dense, trains the final `MLPClassifier` (128-64-32, ReLU, adam, alpha = 0.05, early stopping patience 20), evaluates on the holdout fold, computes the confusion matrix and social cost, prints a classification report and the `[REPORT NUMBERS]` block, and saves the heatmap, loss curve, audit CSV, and pickled model.

**Findings.**

1. **Efficiency.** `X_train.toarray()` densifies a ~63,000 × ~2,300 sparse matrix into roughly 1.2 GB of float64. `MLPClassifier` accepts scipy sparse input directly; the conversion is unnecessary and is the peak memory point of the whole pipeline. Same issue in `03_nn_tuning.py` and `06_confidence_analysis.py`.
2. **Coupling.** `ALPHA = 0.05` is the winner from `03_nn_tuning.py`, copied in by hand. If tuning is re-run and a different alpha wins, nothing updates this constant. Reading the best row from `nn_tuning_results.csv` would close the loop. The README run order (tuning before final training) documents the dependency, which is the minimum acceptable handling.
3. **Implicit hyperparameter.** The tuned best learning rate was 0.001, and this script uses it only because 0.001 happens to be sklearn's default `learning_rate_init`. Nothing in this file says so. One explicit line (`learning_rate_init=0.001`) would make the tuned configuration visible.
4. **Config drift between tuning and final training.** Tuning used `max_iter=300, n_iter_no_change=15`; the final model uses `max_iter=500, n_iter_no_change=20`. The selected alpha was therefore chosen under a slightly different training budget than the one used for the final model. The report discloses both settings, but does not point out that they differ. In practice alpha rankings are stable across such small budget changes; still, tuning and final training should use identical settings.
5. **Cosmetic.** Console step labels jump from `[STEP 4]` to `[STEP 6]`; the social cost audit banner between them is the missing step 5. Numbering only, nothing skipped.
6. **Minor inconsistency.** Lift is computed against the Class 2 rate of the validation fold here, but against the full-dataset rate in `model_audit.py`. Stratification makes the two rates equal to 4 decimal places, so the reported lifts are comparable, but the two scripts should share one definition.
7. **Dead code.** The NaN check after loading (`if np.any(np.isnan(X_train))`) can never fire — `01_nn_data_prep.py` step 8 already verified no NaN in the same arrays. Harmless defensive code.
8. **Good practice worth keeping.** The `[REPORT NUMBERS]` block prints the confusion matrix and every derived metric from the single run, labeled as such. This is exactly what prevents the matrix/metric mismatch described in Section 3.3.

### 4.7 `src/03_nn_tuning.py` (Step 6 — MLP grid search)

**What it does.** A hand-written grid search over 5 alphas × 2 learning rates (10 fits), architecture fixed at 128-64-32, each fit trained from scratch, ranked by validation social cost. Writes `nn_tuning_results.csv`. A custom loop is used instead of `GridSearchCV` because the selection criterion is social cost, not accuracy — the report explains this and the reasoning is valid.

**Findings.**

1. **Interpretability of one output column.** `train_social_cost` is computed on the **oversampled** training fold, where every Class 2 pump is counted 5 times, so its FN component is inflated 5× relative to the validation figure. The printed `Gap` column therefore mixes two different scales and its sign and magnitude do not mean what a reader would assume. Selection uses only `val_social_cost`, which is measured correctly, so the chosen configuration is unaffected.
2. **Same dense-conversion cost as 4.6.1, times 10 fits.**
3. **No checkpointing.** A crash on run 9 of 10 loses all results, since the CSV is written only at the end. At ~15 minutes total this is acceptable; appending each row as it finishes would cost two lines.
4. **Inherited.** Each of the 10 fits uses the contaminated internal early-stopping split described in 4.5.2, so the ranking is based on slightly optimistic stopping decisions. All 10 fits are affected the same way, which limits the damage to the comparison.

### 4.8 `src/06_confidence_analysis.py` (Step 8 — TR-04 confidence verification)

**What it does.** Retrains the weight-5 XGBoost model, loads the pickled MLP, validates both models' `predict_proba` output (shape, finite, in [0, 1], rows sum to 1 within 1e-5 — raises on failure), builds a sample prediction table covering high-confidence correct, low-confidence correct, missed Class 2, and wrong predictions, prints confidence statistics, and saves the P(Class 2) distribution histogram.

**Findings.**

1. **Duplication of training.** The XGBoost model is retrained from scratch instead of loading a model saved by `model_audit.py`. Because the parameters, seed, split, and sample weights are identical, the retrained model reproduces the audited one, but that identity is by convention, not enforced. Saving the model in `model_audit.py` and loading it here would make it a fact. Cost of the current approach: ~2 minutes of duplicate compute per run, and a silent divergence if anyone edits the parameters in one file and not the other.
2. **Verified: both models are scored on the same pumps.** This script splits the parquet matrix with the same seed, stratify target, and test fraction as `01_nn_data_prep.py`, so the XGBoost and MLP validation rows are the same physical pumps in the same order. The report's fairness claim in §II.C depends on this and it holds. It holds silently, though — an assert on the index would make it explicit.
3. **Structure.** This is the only script written as top-level module code instead of functions with an `if __name__ == "__main__"` guard. It cannot be imported without executing ~4 minutes of training. Inconsistent with the other seven files.
4. **Fragile sampling.** `sample_indices = list(set(sample_indices))[:8]` deduplicates and truncates the curated example list. Set ordering is stable for integers within one Python build, but the truncation can silently drop the carefully selected categories (for example the missed-Class-2 example) if more than 8 candidates accumulate. Slicing a list after an ordered dedup (`dict.fromkeys`) would preserve the intended coverage.
5. **Duplication with drift, concrete case.** `validate_probability_output` here checks `proba.shape[1] == 3`; the copy in `model_audit.py` checks `proba.shape == (expected_rows, 3)`. Both pass today, but the two TR-04 gates are already not the same check. This is the standard argument for a shared module (see 5.2).
6. **Good practice.** This is the only place in the pipeline where a requirement failure raises an exception instead of printing. TR-04 is treated as a hard gate. Correct.

---

## 5. Cross-cutting findings

### 5.1 Reproducibility: good

`random_state=42` is set at every split, every model, and the oversampling shuffle. Runs are repeatable on one machine with one environment. Two caveats: exact MLP numbers can still shift across sklearn/BLAS versions (which is the likely origin of the report's Section 3.3 mismatch), and `requirements.txt` previously had no version bounds at all — addressed in this repository with minimum versions (`scikit-learn>=1.2` for `sparse_output=`, `xgboost>=1.6` for constructor `early_stopping_rounds`).

### 5.2 Duplication

`calculate_social_cost` exists in 3 files, `validate_probability_output` in 2 (already divergent, see 4.8.5), and `CLASS_LABELS`, `RANDOM_STATE`, `REFERENCE_YEAR`, and the Windows UTF-8 stdout wrapper are repeated in up to 8 files. One `src/common.py` would remove all of it. This was **not** done in this repository because keeping the files identical to the ones the report's appendix and figures reference outweighs the cleanup — but it is the first refactor to make if the project continues.

### 5.3 Global warning suppression

Seven of eight files start with `warnings.filterwarnings('ignore')`. This hides the deprecations that are actually present (pandas `groupby.apply` behavior, XGBoost `use_label_encoder`) along with anything unexpected. For a coursework submission with fixed library versions this is a defensible choice to keep console output clean; for a maintained project the filters should be scoped to the specific known warnings.

### 5.4 Outputs go to the working directory

Every script reads and writes its files relative to the current working directory, so the repository root fills with ~25 generated files after a full run. Kept as-is to preserve behavior; the `.gitignore` in this repository covers all generated filenames so none of them can be committed by accident, and the README states that scripts must be run from the repository root.

### 5.5 No tests

There is no test of any kind. The two functions that carry the report's central claims — `calculate_social_cost` and `validate_probability_output` — are pure functions that can be tested against hand-computed values in ten lines. That is the highest-value test to add first.

### 5.6 Pickle artifacts

Models and preprocessed arrays are stored with `pickle`. Two known limitations: pickles are not portable across library versions, and loading a pickle executes arbitrary code, so no pickle from an untrusted source should ever be loaded. For local single-user use this is fine.

### 5.7 File naming

The scripts are numbered `01_`, `02_`, `03_`, `06_` — the gap comes from experiment scripts (04, 05) that were cut from the focused submission. The names were kept because the report's Appendix A.2 references them exactly; renaming would break the report's reproduction instructions. The README documents the actual execution order, which is the authoritative sequence.

---

## 6. Changes applied in this repository

Every change below is behavior-preserving with respect to the model results. Nothing that affects a trained model, a metric, or a figure was touched.

| File | Change | Reason |
|---|---|---|
| `src/data_cleaning.py` | Replaced 3 hardcoded `C:\Users\...` paths with a search over `data/` and the working directory; raises a clear error naming `data/README.md` if the files are absent. | Anyone can run it; no personal paths in a public repository. |
| `src/data_cleaning.py` | Removed the `- 1` from the printed test row count. | Display bug (printed 14,849 instead of 14,850). |
| `src/strategic_eda.py` | Same path fix for the raw-CSV candidate list. | Same reason. |
| `src/feature_engineering.py` | Deleted the empty `PHASE 2: XGBOOST PIPELINE` banner at end of file. | Dead leftover from the pre-split combined script. |
| `requirements.txt` | Added minimum versions to all 8 dependencies. | The code uses APIs that do not exist in older releases (see 5.1). |

## 7. Found but deliberately not changed

These are the real improvement items. They were left alone because fixing them would change the trained models or the printed numbers, and the numbers in `docs/technical_report.pdf` were produced by this exact code. Listed in the order they are worth doing:

1. **Add an untouched test set** so that XGBoost early stopping and weight selection stop consuming the reported validation set (4.4.1).
2. **Keep the MLP's internal early-stopping split free of oversampled duplicates** (4.5.2), then re-run tuning with the same `max_iter`/patience as the final model (4.6.4).
3. **Extract `src/common.py`** for the social cost function, probability validator, labels, and constants (5.2), and add the two ten-line unit tests (5.5).
4. **Keep matrices sparse end to end**: `sparse_output=True` in feature engineering, no `toarray()` before the MLP (4.3.1, 4.6.1).
5. **Save the audited XGBoost model and load it in the confidence analysis** instead of retraining (4.8.1).
6. **Regenerate all MLP numbers in the report from one run** using the `[REPORT NUMBERS]` block, fixing the off-by-one social cost (2,878 vs 2,879), the accuracy (0.7494 vs 0.7501), the Class 1 precision (0.8318 vs 0.8351), and the "507 versus 551" sentence (513 is the matrix value) (3.3).
7. Smaller items: raise on unmapped targets (4.1.4), make an explicit decision about `date_recorded` (4.1.5), compute the median construction year instead of hardcoding 2000 (4.3.2), stop building the dropped `pump_age_bucket` (4.3.3), assert engineered flags are nonzero (4.3.4), set `learning_rate_init` explicitly (4.6.3), share one lift definition (4.6.6), wrap `06_confidence_analysis.py` in a main function (4.8.3), and scope the warning filters (5.3).
