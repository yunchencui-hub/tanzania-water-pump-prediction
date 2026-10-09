# Tanzania Water Pump Predictive Maintenance

Cost-sensitive machine learning for predicting the operational status of rural water pumps in Tanzania, built on the DrivenData ["Pump it Up"](https://www.drivendata.org/competitions/7/pump-it-up-data-mining-the-water-table/) dataset. The project compares XGBoost against a Multi-Layer Perceptron (MLP) under a custom social cost objective, and selects the model that minimizes the real-world cost of mistakes instead of maximizing accuracy.

This is the code behind my DT2 technical report: [`docs/technical_report.pdf`](docs/technical_report.pdf). A full line-by-line review of this codebase is in [`docs/CODE_REVIEW.md`](docs/CODE_REVIEW.md).

## Problem

Tanzania operates over 59,000 rural water extraction points. Given the metadata of a pump (construction year, water source, extraction type, payment structure, GPS location, managing authority), predict its status:

| Class | Label | Share of data |
|---|---|---|
| 0 | Functional | 54.31% |
| 1 | Non-Functional | 38.42% |
| 2 | Needs Repair | 7.27% |

Missing a pump that needs repair costs more than sending a crew to a healthy pump: residents lose access to clean water (UN SDG 6). So evaluation does not use accuracy. It uses a social cost function:

```
Social Cost = 5 × FN(Class 2) + 1 × FP(Class 2)
```

A missed repair counts five times more than a false alarm. The 5:1 ratio follows industry data putting reactive emergency repair at 3–5× the cost of routine maintenance.

## Results

Both models were evaluated on the same stratified 20% holdout (11,880 pumps, `random_state=42`).

| Metric | XGBoost (weight 5×) | MLP (α = 0.05) |
|---|---|---|
| Social Cost | **2,752** | 2,879 |
| Class 2 Recall | 0.6385 | 0.5944 |
| Class 1 Precision | 0.8753 | 0.8351 |
| Accuracy | 0.7507 | 0.7501 |
| Class 2 Lift | 8.78× | 8.18× |
| Overfit Gap | 0.038 | 0.122 |

**XGBoost with a Class 2 sample weight of 5 is the final recommendation.** It has the lowest social cost and is the only configuration that passes all five technical requirements (TR-01 recall > 0.50, TR-02 precision > 0.80, TR-03 overfit gap < 0.10, TR-04 probabilistic output, TR-05 lift > 3.0×). The MLP comes close on cost but fails TR-03, because training on 5× oversampled Class 2 data inflates its training accuracy relative to validation.

The two models handle the 7.27% minority class in deliberately different ways: XGBoost keeps the data untouched and re-weights Class 2 errors in the loss (`sample_weight`); the MLP changes the training distribution itself by physically copying every Class 2 training sample 5 times. Without any weighting, XGBoost misses 74% of the pumps that need repair.

## Repository structure

```
├── README.md
├── LICENSE
├── requirements.txt
├── data/                    # dataset goes here (not committed; see data/README.md)
│   └── README.md
├── docs/
│   ├── technical_report.pdf # full 11-page technical report
│   └── CODE_REVIEW.md       # detailed line-by-line code review
└── src/
    ├── data_cleaning.py           # step 1: clean raw CSVs, encode target
    ├── strategic_eda.py           # step 2: risk-oriented EDA + numerical analysis
    ├── feature_engineering.py     # step 3: risk flags, pump age, one-hot encoding
    ├── model_audit.py             # step 4: XGBoost class-weight sensitivity + TR audit
    ├── 01_nn_data_prep.py         # step 5: MLP scaling, encoding, oversampling
    ├── 03_nn_tuning.py            # step 6: MLP grid search (10 configs, by social cost)
    ├── 02_nn_train.py             # step 7: final MLP training + validation audit
    └── 06_confidence_analysis.py  # step 8: TR-04 probability output verification
```

The script filenames are kept exactly as referenced in Appendix A.2 of the technical report (the `04`/`05` gap comes from experiment scripts that are not part of the assessed pipeline).

## Getting started

### 1. Install

Requires Python 3.10+.

```bash
git clone https://github.com/<yunchencui-hub>/tanzania-water-pump-prediction.git
cd tanzania-water-pump-prediction
pip install -r requirements.txt
```

### 2. Get the data

Download the three CSV files from the [DrivenData competition page](https://www.drivendata.org/competitions/7/pump-it-up-data-mining-the-water-table/) (free account required) and place them in `data/`:

```
data/training_set_values.csv
data/training_set_labels.csv
data/test_set_values.csv
```

Details in [`data/README.md`](data/README.md). The data is not committed to this repository.

### 3. Run the pipeline

Run every script from the repository root, in this order. Intermediate files and plots are written to the working directory.

```bash
python src/data_cleaning.py           # -> cleaned_*.csv
python src/strategic_eda.py           # -> EDA plots (figures 1-7 of the report)
python src/feature_engineering.py     # -> final_X_train.parquet, final_y_train.csv
python src/model_audit.py             # -> audit_results.csv, confusion matrices, feature importance
python src/01_nn_data_prep.py         # -> nn_preprocessed_data.pkl
python src/03_nn_tuning.py            # -> nn_tuning_results.csv (provides alpha for the next step)
python src/02_nn_train.py             # -> nn_model.pkl, nn_audit_results.csv
python src/06_confidence_analysis.py  # -> confidence_samples.csv, confidence_distribution.png
```

Total runtime is about 15 minutes on consumer hardware. All experiments use `random_state=42`.

`03_nn_tuning.py` runs before `02_nn_train.py` because the tuning result (α = 0.05) is the value the final training script uses.

## Design decisions

| Decision | Choice | Reason |
|---|---|---|
| Evaluation metric | Custom social cost loop | `GridSearchCV` optimizes accuracy, which ignores the 5:1 cost asymmetry |
| Class imbalance | XGBoost: `sample_weight`; MLP: 5× physical oversampling | `MLPClassifier` has no `class_weight`; SMOTE creates unrealistic samples on one-hot data |
| Data leakage | Split first, then fit `StandardScaler`/`OneHotEncoder` on the training fold only | Fitting on all data would leak validation statistics into training |
| Overfitting control | Early stopping (XGBoost: 50 rounds; MLP: 20 epochs) + L2 | Cheaper than k-fold CV at 60K samples |
| Reproducibility | `random_state=42` everywhere | Both models are evaluated on the same physical pumps |

## Documentation

- [`docs/technical_report.pdf`](docs/technical_report.pdf) — the full study: problem definition, requirements, EDA findings, model selection reasoning, sensitivity analysis, confusion matrix analysis, and limitations.
- [`docs/CODE_REVIEW.md`](docs/CODE_REVIEW.md) — a detailed review of this codebase: verification of every reported number against the code, file-by-file findings with locations, and a prioritized improvement list.

## References

1. Grinsztajn, L., Oyallon, E., & Varoquaux, G. (2022). Why do tree-based models still outperform deep learning on typical tabular data? *NeurIPS 35*.
2. Shwartz-Ziv, R., & Armon, A. (2022). Tabular data: Deep learning is not all you need. *Information Fusion, 81*, 84–90.
3. Chawla, N. V. et al. (2002). SMOTE: synthetic minority over-sampling technique. *JAIR, 16*, 321–357.
4. DrivenData. (2015). Pump it Up: Data Mining the Water Table.

## License

[MIT](LICENSE)
