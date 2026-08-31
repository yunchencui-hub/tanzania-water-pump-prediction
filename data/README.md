# Data

The dataset is not committed to this repository. It comes from the DrivenData competition **"Pump it Up: Data Mining the Water Table"**, sourced from the Tanzanian Ministry of Water via the Taarifa waterpoints dashboard.

## How to get it

1. Create a free account at https://www.drivendata.org/
2. Join the competition: https://www.drivendata.org/competitions/7/pump-it-up-data-mining-the-water-table/
3. Download the three files from the competition's data page and save them in this folder with these names:

```
data/training_set_values.csv    # 59,400 pumps × 40 features
data/training_set_labels.csv    # 59,400 labels (status_group)
data/test_set_values.csv        # 14,850 pumps (labels withheld by the competition)
```

The scripts also accept the display names DrivenData uses for the downloads (`Training set values.csv`, `Training set labels.csv`, `Test set values.csv`), placed either in this folder or in the repository root.

## Target variable

`status_group`, mapped by the pipeline to:

| Code | Label | Share |
|---|---|---|
| 0 | functional | 54.31% |
| 1 | non functional | 38.42% |
| 2 | functional needs repair | 7.27% |

## Why the data is not in the repository

The competition's terms require downloading the data through DrivenData, and generated artifacts (cleaned CSVs, Parquet matrices, pickles) are large and fully reproducible by running the pipeline.
