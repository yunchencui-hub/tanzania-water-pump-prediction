# Step 8: MLP Data Preparation: 01_nn_data_prep.py

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import scipy.sparse as sp
import sys
import io
import warnings
import pickle

warnings.filterwarnings('ignore')

if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

HIGH_CARDINALITY_THRESHOLD = 50

CLASS_2_OVERSAMPLE_FACTOR = 5

VAL_SIZE = 0.20
RANDOM_STATE = 42

def identify_column_types(df):
    numerical_cols = df.select_dtypes(include=['int64', 'float64', 'int32', 'float32']).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()
    
    return numerical_cols, categorical_cols

def filter_high_cardinality(df, categorical_cols, threshold):
    valid_cats = []
    dropped_cats = []
    
    for col in categorical_cols:
        n_unique = df[col].nunique()
        if n_unique > threshold:
            dropped_cats.append((col, n_unique))
        else:
            valid_cats.append(col)
    
    return valid_cats, dropped_cats

def physical_oversample_class2(X, y, factor):
    class2_mask = (y == 2)
    X_class2 = X[class2_mask]
    y_class2 = y[class2_mask]
    
    class2_count_before = len(y_class2)
    
    X_oversampled_parts = [X]
    y_oversampled_parts = [y]
    
    for _ in range(factor - 1):
        X_oversampled_parts.append(X_class2)
        y_oversampled_parts.append(y_class2)
    
    if sp.issparse(X):
        X_oversampled = sp.vstack(X_oversampled_parts)
    else:
        X_oversampled = np.vstack(X_oversampled_parts)
    
    y_oversampled = np.concatenate(y_oversampled_parts)
    
    shuffle_idx = np.random.RandomState(RANDOM_STATE).permutation(len(y_oversampled))
    X_oversampled = X_oversampled[shuffle_idx]
    y_oversampled = y_oversampled[shuffle_idx]
    
    class2_count_after = (y_oversampled == 2).sum()
    
    return X_oversampled, y_oversampled, class2_count_before, class2_count_after

def run_nn_data_prep():
    
    print("\n" + "=" * 70)
    print("  NEURAL NETWORK DATA PREPROCESSING PIPELINE")
    print("  Challenger Model Phase - Tanzania Water Pump Challenge")
    print("=" * 70)
    
    print("\n[STEP 1] Loading Data")
    print("-" * 55)
    
    X = pd.read_parquet('final_X_train.parquet')
    y = pd.read_csv('final_y_train.csv')['status_group'].values
    
    print(f"   ✓ Features shape: {X.shape}")
    print(f"   ✓ Labels shape: {y.shape}")
    print(f"   ✓ Class distribution:")
    for c in [0, 1, 2]:
        count = (y == c).sum()
        pct = count / len(y) * 100
        print(f"      Class {c}: {count:,} ({pct:.2f}%)")
    
    print("\n[STEP 2] Identifying Column Types")
    print("-" * 55)
    
    numerical_cols, categorical_cols = identify_column_types(X)
    
    print(f"   ✓ Numerical columns: {len(numerical_cols)}")
    print(f"   ✓ Categorical columns: {len(categorical_cols)}")
    
    print("\n[STEP 3] Filtering High Cardinality Columns (NFR-02)")
    print("-" * 55)
    print(f"   Threshold: > {HIGH_CARDINALITY_THRESHOLD} unique values")
    
    valid_cats, dropped_cats = filter_high_cardinality(X, categorical_cols, HIGH_CARDINALITY_THRESHOLD)
    
    if len(dropped_cats) > 0:
        print(f"\n   ⚠️  Dropped {len(dropped_cats)} high-cardinality columns:")
        for col, n_unique in dropped_cats:
            print(f"      • {col}: {n_unique} unique values")
    else:
        print("   ✓ No high-cardinality columns found")
    
    print(f"\n   ✓ Valid categorical columns: {len(valid_cats)}")
    
    print("\n[STEP 4] Stratified Train/Val Split")
    print("-" * 55)
    
    X_train_df, X_val_df, y_train, y_val = train_test_split(
        X, y,
        test_size=VAL_SIZE,
        stratify=y,
        random_state=RANDOM_STATE
    )
    
    print(f"   ✓ X_train: {X_train_df.shape}")
    print(f"   ✓ X_val: {X_val_df.shape}")
    print(f"   ✓ y_train: {y_train.shape}")
    print(f"   ✓ y_val: {y_val.shape}")
    
    print("\n[STEP 5] Building Preprocessing Pipeline (FR-03)")
    print("-" * 55)
    
    cols_to_drop = [col for col, _ in dropped_cats]
    X_train_df = X_train_df.drop(columns=cols_to_drop, errors='ignore')
    X_val_df = X_val_df.drop(columns=cols_to_drop, errors='ignore')
    
    numerical_cols, categorical_cols = identify_column_types(X_train_df)
    categorical_cols = [c for c in categorical_cols if c in valid_cats]
    
    print(f"   Final feature columns:")
    print(f"      Numerical: {len(numerical_cols)}")
    print(f"      Categorical: {len(categorical_cols)}")
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_cols),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=True), categorical_cols)
        ],
        remainder='drop',
        sparse_threshold=0.3
    )
    
    print("   ✓ StandardScaler for numerical features")
    print("   ✓ OneHotEncoder for categorical features")
    
    print("\n[STEP 6] Fitting Transformer on Training Data")
    print("-" * 55)
    
    X_train_processed = preprocessor.fit_transform(X_train_df)
    X_val_processed = preprocessor.transform(X_val_df)
    
    print(f"   ✓ X_train processed shape: {X_train_processed.shape}")
    print(f"   ✓ X_val processed shape: {X_val_processed.shape}")
    print(f"   ✓ Sparse matrix: {sp.issparse(X_train_processed)}")
    
    if sp.issparse(X_train_processed):
        density = X_train_processed.nnz / (X_train_processed.shape[0] * X_train_processed.shape[1])
        print(f"   ✓ Matrix density: {density:.4f} ({density*100:.2f}%)")
    
    print("\n[STEP 7] Physical Oversampling for Class 2 (FR-01)")
    print("-" * 55)
    print(f"   Oversampling factor: {CLASS_2_OVERSAMPLE_FACTOR}x")
    print(f"   Objective: Minimize 5*FN + 1*FP")
    
    X_train_oversampled, y_train_oversampled, before, after = physical_oversample_class2(
        X_train_processed, y_train, CLASS_2_OVERSAMPLE_FACTOR
    )
    
    print(f"\n   📊 CLASS 2 VERIFICATION:")
    print(f"      Before oversampling: {before:,} samples")
    print(f"      After oversampling:  {after:,} samples")
    print(f"      Multiplier achieved: {after / before:.1f}x")
    
    print(f"\n   📊 FINAL TRAINING SET DISTRIBUTION:")
    for c in [0, 1, 2]:
        count = (y_train_oversampled == c).sum()
        pct = count / len(y_train_oversampled) * 100
        print(f"      Class {c}: {count:,} ({pct:.2f}%)")
    
    print("\n[STEP 8] Final Data Verification")
    print("-" * 55)
    
    print(f"   ✓ X_train_final shape: {X_train_oversampled.shape}")
    print(f"   ✓ y_train_final shape: {y_train_oversampled.shape}")
    print(f"   ✓ X_val_final shape: {X_val_processed.shape}")
    print(f"   ✓ y_val_final shape: {y_val.shape}")
    
    if sp.issparse(X_train_oversampled):
        has_nan = np.any(np.isnan(X_train_oversampled.data))
        has_inf = np.any(np.isinf(X_train_oversampled.data))
    else:
        has_nan = np.any(np.isnan(X_train_oversampled))
        has_inf = np.any(np.isinf(X_train_oversampled))
    
    print(f"   ✓ Contains NaN: {has_nan}")
    print(f"   ✓ Contains Inf: {has_inf}")
    
    print("\n[STEP 9] Saving Preprocessed Data")
    print("-" * 55)
    
    output_data = {
        'X_train': X_train_oversampled,
        'y_train': y_train_oversampled,
        'X_val': X_val_processed,
        'y_val': y_val,
        'preprocessor': preprocessor,
        'numerical_cols': numerical_cols,
        'categorical_cols': categorical_cols,
        'config': {
            'high_cardinality_threshold': HIGH_CARDINALITY_THRESHOLD,
            'oversample_factor': CLASS_2_OVERSAMPLE_FACTOR,
            'random_state': RANDOM_STATE
        }
    }
    
    with open('nn_preprocessed_data.pkl', 'wb') as f:
        pickle.dump(output_data, f)
    
    print("   ✓ Saved: nn_preprocessed_data.pkl")
    print("   Contains: X_train, y_train, X_val, y_val, preprocessor, config")
    
    print("\n" + "=" * 70)
    print("  DATA PREPROCESSING COMPLETE")
    print("=" * 70)
    print(f"   Final training shape: {X_train_oversampled.shape}")
    print(f"   Final validation shape: {X_val_processed.shape}")
    print(f"   Class 2 oversampling factor: {CLASS_2_OVERSAMPLE_FACTOR}x")
    return output_data


if __name__ == "__main__":
    run_nn_data_prep()
