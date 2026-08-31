# Step 1: Data Cleaning: data_cleaning.py

import pandas as pd
import numpy as np
from typing import Tuple
import sys
import io

if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

COLUMNS_TO_DROP = [
    'scheme_name',
    'quantity_group',
    'num_private',
    'wpt_name',
]

PHYSICS_INVALID_ZERO_COLS = ['gps_height', 'longitude']

KEEP_ZERO_COLS = ['construction_year', 'amount_tsh', 'population']

HIGH_CARDINALITY_COLS = ['installer', 'funder']
TOP_N_CATEGORIES = 20

FILL_UNKNOWN_COLS = ['scheme_management', 'public_meeting', 'permit']

CONVERT_TO_STRING_COLS = ['region_code', 'district_code']

TARGET_ENCODING = {
    'functional': 0,
    'non functional': 1,
    'functional needs repair': 2
}

def reduce_cardinality(
    df: pd.DataFrame,
    column: str,
    top_n: int = TOP_N_CATEGORIES,
    other_label: str = 'Other',
    train_mask: pd.Series = None
) -> pd.DataFrame:
    df = df.copy()
    
    if train_mask is not None:
        top_categories = df.loc[train_mask, column].value_counts().head(top_n).index.tolist()
    else:
        top_categories = df[column].value_counts().head(top_n).index.tolist()
    
    df[column] = df[column].apply(
        lambda x: x if x in top_categories else other_label
    )
    
    return df

def print_cardinality_summary(df: pd.DataFrame, column: str, stage: str = "After") -> None:
    print(f"\n   {stage} - '{column}' categories:")
    value_counts = df[column].value_counts()
    print(f"   Total unique: {len(value_counts)}")
    print(f"   Top 5 categories:")
    for cat, count in value_counts.head(5).items():
        pct = count / len(df) * 100
        print(f"      {cat}: {count:,} ({pct:.2f}%)")
    if 'Other' in value_counts.index:
        other_count = value_counts['Other']
        other_pct = other_count / len(df) * 100
        print(f"   'Other' bucket: {other_count:,} ({other_pct:.2f}%)")

def clean_data(
    train_values: pd.DataFrame,
    train_labels: pd.DataFrame,
    test_values: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    
    print("\n" + "=" * 70)
    print("  SURGICAL DATA CLEANING PIPELINE - Tanzania Water Pump Challenge")
    print("=" * 70)
    
    print("\n[STEP 1] Data Merge & Alignment")
    print("-" * 50)
    
    train_df = train_values.merge(train_labels, on='id')
    
    print(f"   Train Values shape: {train_values.shape}")
    print(f"   Train Labels shape: {train_labels.shape}")
    print(f"   Merged Train shape: {train_df.shape}")
    print(f"   Test Values shape:  {test_values.shape}")
    
    train_ids = train_df['id'].copy()
    test_ids = test_values['id'].copy()
    
    y_train = train_df['status_group'].copy()
    train_df = train_df.drop(columns=['status_group'])
    
    train_df['_is_train'] = True
    test_values = test_values.copy()
    test_values['_is_train'] = False
    
    combined_df = pd.concat([train_df, test_values], axis=0, ignore_index=True)
    
    print(f"   Combined shape:     {combined_df.shape}")
    
    print("\n[STEP 2] Surgical Drops (Redundancy Removal)")
    print("-" * 50)
    
    if 'id' in combined_df.columns:
        combined_df = combined_df.drop(columns=['id'])
        print("   ✓ Dropped 'id' (kept separately for output)")
    
    cols_to_drop_existing = [c for c in COLUMNS_TO_DROP if c in combined_df.columns]
    combined_df = combined_df.drop(columns=cols_to_drop_existing)
    
    for col in cols_to_drop_existing:
        print(f"   ✓ Dropped '{col}'")
    
    print(f"   Shape after drops: {combined_df.shape}")
    
    print("\n[STEP 3] Physics Validity Correction")
    print("-" * 50)
    print("   Replacing 0 with NaN for physically impossible values...")
    
    for col in PHYSICS_INVALID_ZERO_COLS:
        if col in combined_df.columns:
            zero_count_before = (combined_df[col] == 0).sum()
            zero_pct = zero_count_before / len(combined_df) * 100
            
            combined_df[col] = combined_df[col].replace(0, np.nan)
            
            nan_count_after = combined_df[col].isna().sum()
            print(f"   ✓ {col}: {zero_count_before:,} zeros → NaN ({zero_pct:.2f}%)")
    
    print("\n[STEP 4] Informative Missingness Preservation")
    print("-" * 50)
    print("   Keeping 0 values as risk signals (DO NOT IMPUTE)...")
    
    for col in KEEP_ZERO_COLS:
        if col in combined_df.columns:
            zero_count = (combined_df[col] == 0).sum()
            zero_pct = zero_count / len(combined_df) * 100
            
            print(f"   ✓ {col}: {zero_count:,} zeros KEPT ({zero_pct:.2f}%) → Risk signal")
    
    print("\n[STEP 5] Categorical Cleanup")
    print("-" * 50)
    
    for col in HIGH_CARDINALITY_COLS:
        if col in combined_df.columns:
            na_count = combined_df[col].isna().sum()
            combined_df[col] = combined_df[col].fillna('Unknown')
            print(f"   ✓ {col}: Filled {na_count:,} NaN with 'Unknown'")
    
    print(f"\n   Reducing cardinality (keeping Top {TOP_N_CATEGORIES})...")
    
    for col in HIGH_CARDINALITY_COLS:
        if col in combined_df.columns:
            original_unique = combined_df[col].nunique()
            combined_df = reduce_cardinality(combined_df, col, TOP_N_CATEGORIES,
                                              train_mask=combined_df['_is_train'])
            final_unique = combined_df[col].nunique()
            print(f"   ✓ {col}: {original_unique} → {final_unique} categories")
            
            if col == 'installer':
                print_cardinality_summary(combined_df, col)
    
    print(f"\n   Filling missing categoricals with 'Unknown'...")
    
    for col in FILL_UNKNOWN_COLS:
        if col in combined_df.columns:
            na_count = combined_df[col].isna().sum()
            if na_count > 0:
                combined_df[col] = combined_df[col].fillna('Unknown')
                print(f"   ✓ {col}: Filled {na_count:,} NaN with 'Unknown'")
            else:
                print(f"   ✓ {col}: No missing values")
    
    print(f"\n   Converting numeric codes to string (categorical labels)...")
    
    for col in CONVERT_TO_STRING_COLS:
        if col in combined_df.columns:
            original_dtype = combined_df[col].dtype
            combined_df[col] = combined_df[col].astype(str)
            print(f"   ✓ {col}: {original_dtype} → string")
    
    print("\n[STEP 6] Target Encoding")
    print("-" * 50)
    
    y_train_encoded = y_train.map(TARGET_ENCODING)
    
    print(f"   Encoding mapping:")
    for label, code in TARGET_ENCODING.items():
        count = (y_train == label).sum()
        pct = count / len(y_train) * 100
        print(f"      '{label}' → {code} ({count:,} samples, {pct:.2f}%)")
    
    unmapped = y_train_encoded.isna().sum()
    if unmapped > 0:
        print(f"   ⚠️ WARNING: {unmapped} unmapped target values!")
    else:
        print(f"   ✓ All target values encoded successfully")
    
    print("\n[STEP 7] Splitting Train/Test")
    print("-" * 50)
    
    X_train_clean = combined_df[combined_df['_is_train'] == True].drop(columns=['_is_train'])
    X_test_clean = combined_df[combined_df['_is_train'] == False].drop(columns=['_is_train'])
    
    X_train_clean = X_train_clean.reset_index(drop=True)
    X_test_clean = X_test_clean.reset_index(drop=True)
    y_train_clean = y_train_encoded.reset_index(drop=True)
    
    print(f"   X_train_clean shape: {X_train_clean.shape}")
    print(f"   y_train_clean shape: {y_train_clean.shape}")
    print(f"   X_test_clean shape:  {X_test_clean.shape}")
    
    if 'status_group' in X_test_clean.columns:
        print("   ⚠️ WARNING: Target leakage detected in test set!")
    else:
        print("   ✓ No target leakage in test set")
    
    print("\n" + "=" * 70)
    print("  CLEANING COMPLETE - Summary")
    print("=" * 70)
    
    print(f"\n   📊 Dataset Shapes:")
    print(f"      Before: Train ({train_values.shape[0]:,} × {train_values.shape[1]})")
    print(f"      After:  Train ({X_train_clean.shape[0]:,} × {X_train_clean.shape[1]})")
    print(f"      Before: Test  ({test_values.shape[0]:,} × {test_values.shape[1] - 1})")
    print(f"      After:  Test  ({X_test_clean.shape[0]:,} × {X_test_clean.shape[1]})")
    
    print(f"\n   📋 Columns removed: {len(COLUMNS_TO_DROP) + 1}")
    print(f"   📋 Final feature count: {X_train_clean.shape[1]}")
    
    print(f"\n   📊 Data Types:")
    print(f"      Numeric:     {X_train_clean.select_dtypes(include=[np.number]).shape[1]}")
    print(f"      Categorical: {X_train_clean.select_dtypes(include=['object']).shape[1]}")
    
    train_missing = X_train_clean.isna().sum().sum()
    test_missing = X_test_clean.isna().sum().sum()
    print(f"\n   📊 Remaining Missing Values:")
    print(f"      Train: {train_missing:,} ({train_missing / X_train_clean.size * 100:.2f}%)")
    print(f"      Test:  {test_missing:,} ({test_missing / X_test_clean.size * 100:.2f}%)")
    
    print("\n" + "=" * 70)
    print("  Ready for Strategic EDA and Feature Engineering!")
    print("=" * 70 + "\n")
    
    return X_train_clean, y_train_clean, X_test_clean

def _find_data_file(candidates):
    import os
    for path in candidates:
        if os.path.exists(path):
            return path
    raise FileNotFoundError(
        f"Could not find any of: {candidates}. "
        "Download the dataset from DrivenData (see data/README.md) "
        "and place the CSV files in the data/ folder."
    )

if __name__ == "__main__":

    TRAIN_VALUES_PATH = _find_data_file([
        'data/training_set_values.csv',
        'data/Training set values.csv',
        'training_set_values.csv',
        'Training set values.csv',
    ])
    TRAIN_LABELS_PATH = _find_data_file([
        'data/training_set_labels.csv',
        'data/Training set labels.csv',
        'training_set_labels.csv',
        'Training set labels.csv',
    ])
    TEST_VALUES_PATH = _find_data_file([
        'data/test_set_values.csv',
        'data/Test set values.csv',
        'test_set_values.csv',
        'Test set values.csv',
    ])

    print("Loading datasets...")
    
    try:
        train_values = pd.read_csv(TRAIN_VALUES_PATH)
        train_labels = pd.read_csv(TRAIN_LABELS_PATH)
        test_values = pd.read_csv(TEST_VALUES_PATH)
        
        print(f"   ✓ Train Values: {train_values.shape}")
        print(f"   ✓ Train Labels: {train_labels.shape}")
        print(f"   ✓ Test Values:  {test_values.shape}")
        
        X_train_clean, y_train_clean, X_test_clean = clean_data(
            train_values, train_labels, test_values
        )
        
        print("\n[STEP 8] Saving Cleaned Datasets to CSV")
        print("-" * 50)
        
        TRAIN_VALUES_OUTPUT = 'cleaned_train_values.csv'
        TRAIN_LABELS_OUTPUT = 'cleaned_train_labels.csv'
        TEST_VALUES_OUTPUT = 'cleaned_test_values.csv'
        
        X_train_clean.to_csv(TRAIN_VALUES_OUTPUT, index=False)
        print(f"   ✓ Saved: {TRAIN_VALUES_OUTPUT} ({X_train_clean.shape[0]:,} rows × {X_train_clean.shape[1]} cols)")
        
        y_train_df = pd.DataFrame({'status_group': y_train_clean})
        y_train_df.to_csv(TRAIN_LABELS_OUTPUT, index=False)
        print(f"   ✓ Saved: {TRAIN_LABELS_OUTPUT} ({len(y_train_clean):,} rows)")
        
        X_test_clean.to_csv(TEST_VALUES_OUTPUT, index=False)
        print(f"   ✓ Saved: {TEST_VALUES_OUTPUT} ({X_test_clean.shape[0]:,} rows × {X_test_clean.shape[1]} cols)")
        
        print("\n" + "=" * 70)
        print("  All cleaned datasets saved successfully!")
        print("=" * 70 + "\n")
        
    except FileNotFoundError as e:
        print(f"Error: Could not find dataset file - {e}")
        print("Please check file paths and ensure datasets are downloaded.")
