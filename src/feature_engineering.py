# Step 4: Feature Engineering: feature_engineering.py

import pandas as pd
from sklearn.preprocessing import OneHotEncoder
import numpy as np
from typing import Tuple, List
import sys
import io
import warnings

warnings.filterwarnings('ignore')

if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

REFERENCE_YEAR = 2014

MEDIAN_CONSTRUCTION_YEAR = 2000

HOTSPOT_REGIONS = ['Lindi', 'Mtwara']

CATEGORICAL_COLS_FOR_ENCODING = [
    'funder', 'installer', 'basin', 'region', 'lga', 'ward',
    'public_meeting', 'scheme_management', 'permit', 'extraction_type',
    'extraction_type_group', 'extraction_type_class', 'management',
    'management_group', 'payment', 'payment_type', 'water_quality',
    'quality_group', 'quantity', 'source', 'source_type', 'source_class',
    'waterpoint_type', 'waterpoint_type_group'
]

COLUMNS_TO_DROP = ['recorded_by']

def create_sure_kill_flags(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    
    df['is_dry_well'] = (df['quantity'] == 'dry').astype(int)
    
    df['is_never_pay'] = (df['payment_type'] == 'never pay').astype(int)
    
    return df

def create_hidden_risk_flags(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    
    df['is_surface_water'] = (df['source_class'] == 'surface').astype(int)
    
    df['is_unknown_source'] = (df['source_class'] == 'unknown').astype(int)
    
    return df

def create_zero_value_flags(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    
    df['tsh_is_zero'] = (df['amount_tsh'] == 0).astype(int)
    
    df['year_is_missing'] = (df['construction_year'] == 0).astype(int)
    
    df['population_is_zero'] = (df['population'] == 0).astype(int)
    
    df['zero_signal_count'] = (
        df['tsh_is_zero'] + 
        df['year_is_missing'] + 
        df['population_is_zero']
    )
    
    return df

def create_time_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    
    construction_year_imputed = df['construction_year'].copy()
    
    construction_year_imputed = construction_year_imputed.replace(0, MEDIAN_CONSTRUCTION_YEAR)
    
    df['pump_age'] = REFERENCE_YEAR - construction_year_imputed
    
    df['pump_age_bucket'] = pd.cut(
        df['pump_age'],
        bins=[0, 5, 10, 15, 20, 30, 100],
        labels=['0-5', '6-10', '11-15', '16-20', '21-30', '30+']
    )
    
    df['is_old_pump'] = (df['pump_age'] > 20).astype(int)
    
    df['is_very_old_pump'] = (df['pump_age'] > 30).astype(int)
    
    return df

def create_geospatial_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    
    df['is_hotspot_region'] = df['region'].isin(HOTSPOT_REGIONS).astype(int)
    
    return df

def create_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    
    df['sure_kill_score'] = (
        df['is_dry_well'] + 
        df['is_never_pay'] + 
        df['is_hotspot_region']
    )
    
    df['old_pump_in_hotspot'] = (
        df['is_old_pump'] * df['is_hotspot_region']
    ).astype(int)
    
    df['dry_never_pay'] = (
        df['is_dry_well'] * df['is_never_pay']
    ).astype(int)
    
    df['compound_risk_score'] = (
        df['sure_kill_score'] + 
        df['zero_signal_count'] +
        df['is_old_pump'] +
        df['is_surface_water']
    )
    
    return df

def encode_categoricals(
    df: pd.DataFrame,
    categorical_cols: List[str],
    train_mask: pd.Series = None
) -> pd.DataFrame:
    df = df.copy()
    
    cols_to_encode = [c for c in categorical_cols if c in df.columns]
    
    existing_cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    cols_to_encode = [c for c in cols_to_encode if c in existing_cat_cols]
    
    if len(cols_to_encode) > 0:
        for c in cols_to_encode:
            df[c] = df[c].fillna('_missing_')
        
        enc = OneHotEncoder(handle_unknown='ignore', sparse_output=False, dtype=int)
        
        if train_mask is not None:
            enc.fit(df.loc[train_mask, cols_to_encode])
        else:
            enc.fit(df[cols_to_encode])
        
        encoded = enc.transform(df[cols_to_encode])
        encoded_cols = enc.get_feature_names_out(cols_to_encode)
        encoded_df = pd.DataFrame(encoded, columns=encoded_cols, index=df.index)
        
        df = df.drop(columns=cols_to_encode)
        df = pd.concat([df, encoded_df], axis=1)
    
    return df

def cleanup_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    
    for col in COLUMNS_TO_DROP:
        if col in df.columns:
            df = df.drop(columns=[col])
    
    if 'pump_age_bucket' in df.columns:
        df = df.drop(columns=['pump_age_bucket'])
    
    return df

def engineer_features(
    train_values_path: str,
    train_labels_path: str,
    test_values_path: str
) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    
    print("\n" + "=" * 70)
    print("  FEATURE ENGINEERING PIPELINE - Tanzania Water Pump Challenge")
    print("=" * 70)
    
    print("\n[STEP 1] Loading & Concatenating Data")
    print("-" * 50)
    
    train_values = pd.read_csv(
        train_values_path,
        dtype={'region_code': str, 'district_code': str}
    )
    train_labels = pd.read_csv(train_labels_path)
    test_values = pd.read_csv(
        test_values_path,
        dtype={'region_code': str, 'district_code': str}
    )
    
    n_train = len(train_values)
    n_test = len(test_values)
    
    print(f"   ✓ Train Values: {train_values.shape}")
    print(f"   ✓ Train Labels: {train_labels.shape}")
    print(f"   ✓ Test Values:  {test_values.shape}")
    
    train_values['_is_train'] = True
    test_values['_is_train'] = False
    
    df = pd.concat([train_values, test_values], axis=0, ignore_index=True)
    
    print(f"   ✓ Combined:     {df.shape}")
    
    print("\n[STEP 2] Creating Sure-Kill Binary Flags")
    print("-" * 50)
    
    df = create_sure_kill_flags(df)
    
    print(f"   ✓ is_dry_well:   {df['is_dry_well'].sum():,} pumps ({df['is_dry_well'].mean()*100:.2f}%)")
    print(f"   ✓ is_never_pay:  {df['is_never_pay'].sum():,} pumps ({df['is_never_pay'].mean()*100:.2f}%)")
    
    print("\n[STEP 3] Creating Hidden Risk Flags")
    print("-" * 50)
    
    df = create_hidden_risk_flags(df)
    
    print(f"   ✓ is_surface_water:  {df['is_surface_water'].sum():,} pumps ({df['is_surface_water'].mean()*100:.2f}%)")
    print(f"   ✓ is_unknown_source: {df['is_unknown_source'].sum():,} pumps ({df['is_unknown_source'].mean()*100:.2f}%)")
    
    print("\n[STEP 4] Creating Zero-Value Risk Signals")
    print("-" * 50)
    
    df = create_zero_value_flags(df)
    
    print(f"   ✓ tsh_is_zero:        {df['tsh_is_zero'].sum():,} pumps ({df['tsh_is_zero'].mean()*100:.2f}%)")
    print(f"   ✓ year_is_missing:    {df['year_is_missing'].sum():,} pumps ({df['year_is_missing'].mean()*100:.2f}%)")
    print(f"   ✓ population_is_zero: {df['population_is_zero'].sum():,} pumps ({df['population_is_zero'].mean()*100:.2f}%)")
    print(f"   ✓ zero_signal_count:  mean = {df['zero_signal_count'].mean():.2f}")
    
    print("\n[STEP 5] Creating Time-Based Features (Pump Age)")
    print("-" * 50)
    
    df = create_time_features(df)
    
    print(f"   ✓ pump_age:       mean = {df['pump_age'].mean():.1f} years, range = [{df['pump_age'].min():.0f}, {df['pump_age'].max():.0f}]")
    print(f"   ✓ is_old_pump:    {df['is_old_pump'].sum():,} pumps ({df['is_old_pump'].mean()*100:.2f}%)")
    print(f"   ✓ is_very_old_pump: {df['is_very_old_pump'].sum():,} pumps ({df['is_very_old_pump'].mean()*100:.2f}%)")
    print(f"   ✓ Imputed construction_year=0 with median ({MEDIAN_CONSTRUCTION_YEAR})")
    
    print("\n[STEP 6] Creating Geospatial Hotspot Features")
    print("-" * 50)
    
    df = create_geospatial_features(df)
    
    print(f"   ✓ is_hotspot_region: {df['is_hotspot_region'].sum():,} pumps ({df['is_hotspot_region'].mean()*100:.2f}%)")
    print(f"   ✓ Hotspot regions: {HOTSPOT_REGIONS}")
    
    print("\n[STEP 7] Creating Interaction Features")
    print("-" * 50)
    
    df = create_interaction_features(df)
    
    print(f"   ✓ sure_kill_score:      mean = {df['sure_kill_score'].mean():.2f}, max = {df['sure_kill_score'].max()}")
    print(f"   ✓ compound_risk_score:  mean = {df['compound_risk_score'].mean():.2f}, max = {df['compound_risk_score'].max()}")
    print(f"   ✓ old_pump_in_hotspot:  {df['old_pump_in_hotspot'].sum():,} pumps")
    print(f"   ✓ dry_never_pay:        {df['dry_never_pay'].sum():,} pumps")
    
    engineered_features = [
        'is_dry_well', 'is_never_pay', 'is_surface_water', 'is_unknown_source',
        'tsh_is_zero', 'year_is_missing', 'population_is_zero', 'zero_signal_count',
        'pump_age', 'is_old_pump', 'is_very_old_pump',
        'is_hotspot_region', 'sure_kill_score', 'compound_risk_score',
        'old_pump_in_hotspot', 'dry_never_pay'
    ]
    
    print("\n[STEP 8] One-Hot Encoding Categorical Variables")
    print("-" * 50)
    
    n_cols_before = df.shape[1]
    df = encode_categoricals(df, CATEGORICAL_COLS_FOR_ENCODING,
                              train_mask=df['_is_train'])
    n_cols_after = df.shape[1]
    
    print(f"   ✓ Columns before encoding: {n_cols_before}")
    print(f"   ✓ Columns after encoding:  {n_cols_after}")
    print(f"   ✓ New one-hot columns:     {n_cols_after - n_cols_before}")
    
    print("\n[STEP 9] Final Cleanup")
    print("-" * 50)
    
    df = cleanup_features(df)
    
    print(f"   ✓ Final shape: {df.shape}")
    
    print("\n[STEP 10] Splitting Train/Test")
    print("-" * 50)
    
    X_train = df[df['_is_train'] == True].drop(columns=['_is_train'])
    X_test = df[df['_is_train'] == False].drop(columns=['_is_train'])
    
    X_train = X_train.reset_index(drop=True)
    X_test = X_test.reset_index(drop=True)
    
    y_train = train_labels['status_group']
    
    print(f"   ✓ X_train shape: {X_train.shape}")
    print(f"   ✓ y_train shape: {y_train.shape}")
    print(f"   ✓ X_test shape:  {X_test.shape}")
    
    assert len(X_train) == n_train, "Train size mismatch!"
    assert len(X_test) == n_test, "Test size mismatch!"
    assert len(y_train) == n_train, "Label size mismatch!"
    
    print(f"   ✓ Data alignment verified")
    
    print("\n" + "=" * 70)
    print("  FEATURE ENGINEERING COMPLETE - Summary")
    print("=" * 70)
    
    print(f"\n   📊 Final Dataset Shapes:")
    print(f"      X_train: ({X_train.shape[0]:,} rows × {X_train.shape[1]} features)")
    print(f"      y_train: ({len(y_train):,} labels)")
    print(f"      X_test:  ({X_test.shape[0]:,} rows × {X_test.shape[1]} features)")
    
    print(f"\n   📋 New Engineered Features ({len(engineered_features)}):")
    for feat in engineered_features:
        print(f"      • {feat}")
    
    print(f"\n   📊 Feature Types:")
    print(f"      Numeric:  {X_train.select_dtypes(include=[np.number]).shape[1]}")
    print(f"      Boolean:  {X_train.select_dtypes(include=['bool']).shape[1]}")
    
    train_missing = X_train.isna().sum().sum()
    test_missing = X_test.isna().sum().sum()
    print(f"\n   📊 Missing Values:")
    print(f"      Train: {train_missing:,}")
    print(f"      Test:  {test_missing:,}")
    
    return X_train, y_train, X_test, engineered_features

if __name__ == "__main__":
    
    TRAIN_VALUES_PATH = 'cleaned_train_values.csv'
    TRAIN_LABELS_PATH = 'cleaned_train_labels.csv'
    TEST_VALUES_PATH = 'cleaned_test_values.csv'
    
    OUTPUT_X_TRAIN_PARQUET = 'final_X_train.parquet'
    OUTPUT_Y_TRAIN_CSV = 'final_y_train.csv'
    OUTPUT_X_TEST_PARQUET = 'final_X_test.parquet'
    
    OUTPUT_X_TRAIN_PICKLE = 'final_X_train.pkl'
    OUTPUT_X_TEST_PICKLE = 'final_X_test.pkl'
    
    print("Starting Feature Engineering Pipeline...")
    
    try:
        X_train, y_train, X_test, engineered_features = engineer_features(
            TRAIN_VALUES_PATH,
            TRAIN_LABELS_PATH,
            TEST_VALUES_PATH
        )
        
        print("\n[SAVING] Writing Final ML-Ready Datasets")
        print("-" * 50)
        print("   Using Parquet format for large datasets (faster & smaller)")
        
        X_train.to_parquet(OUTPUT_X_TRAIN_PARQUET, index=False, engine='pyarrow')
        print(f"   ✓ Saved: {OUTPUT_X_TRAIN_PARQUET} ({X_train.shape[0]:,} × {X_train.shape[1]})")
        
        X_test.to_parquet(OUTPUT_X_TEST_PARQUET, index=False, engine='pyarrow')
        print(f"   ✓ Saved: {OUTPUT_X_TEST_PARQUET} ({X_test.shape[0]:,} × {X_test.shape[1]})")
        
        y_train_df = pd.DataFrame({'status_group': y_train})
        y_train_df.to_csv(OUTPUT_Y_TRAIN_CSV, index=False)
        print(f"   ✓ Saved: {OUTPUT_Y_TRAIN_CSV} ({len(y_train):,} labels)")
        
        print("\n   Saving pickle backups...")
        X_train.to_pickle(OUTPUT_X_TRAIN_PICKLE)
        print(f"   ✓ Saved: {OUTPUT_X_TRAIN_PICKLE}")
        
        X_test.to_pickle(OUTPUT_X_TEST_PICKLE)
        print(f"   ✓ Saved: {OUTPUT_X_TEST_PICKLE}")
        
        print("\n" + "=" * 70)
        print("  ML-READY MATRICES GENERATED SUCCESSFULLY!")
        print("=" * 70)
        print(f"\n  Final Train Shape: ({X_train.shape[0]:,}, {X_train.shape[1]})")
        print(f"  Final Test Shape:  ({X_test.shape[0]:,}, {X_test.shape[1]})")
        print("\n  Files saved:")
        print(f"    • Parquet (recommended): {OUTPUT_X_TRAIN_PARQUET}, {OUTPUT_X_TEST_PARQUET}")
        print(f"    • Pickle (fastest load):  {OUTPUT_X_TRAIN_PICKLE}, {OUTPUT_X_TEST_PICKLE}")
        print(f"    • Labels (CSV):           {OUTPUT_Y_TRAIN_CSV}")
        print("\n  Load with:")
        print("    X_train = pd.read_parquet('final_X_train.parquet')")
        print("    X_train = pd.read_pickle('final_X_train.pkl')")
        print("\n  Ready for model training!")
        print("=" * 70 + "\n")
        
    except FileNotFoundError as e:
        print(f"Error: Could not find file - {e}")
        print("Make sure you have run data_cleaning.py first.")
    except ImportError:
        print("Error: pyarrow not installed. Install with: pip install pyarrow")
        print("Falling back to pickle only...")
        X_train.to_pickle(OUTPUT_X_TRAIN_PICKLE)
        X_test.to_pickle(OUTPUT_X_TEST_PICKLE)
        y_train_df = pd.DataFrame({'status_group': y_train})
        y_train_df.to_csv(OUTPUT_Y_TRAIN_CSV, index=False)
        print("Pickle files saved successfully.")
