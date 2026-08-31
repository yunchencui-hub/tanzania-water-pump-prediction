# Step 10: MLP Hyperparameter Tuning: 03_nn_tuning.py

import numpy as np
import pandas as pd
import pickle
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
import scipy.sparse as sp
import sys
import io
import time
import warnings
from itertools import product

warnings.filterwarnings('ignore')

if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

HIDDEN_LAYER_SIZES = (128, 64, 32)

ALPHA_VALUES = [0.0001, 0.001, 0.01, 0.05, 0.1]
LEARNING_RATE_VALUES = [0.001, 0.0005]

MAX_ITER = 300
EARLY_STOPPING = True
VALIDATION_FRACTION = 0.1
N_ITER_NO_CHANGE = 15
RANDOM_STATE = 42

def calculate_social_cost(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])
    
    FN_2 = cm[2, :].sum() - cm[2, 2]
    
    FP_2 = cm[:, 2].sum() - cm[2, 2]
    
    social_cost = (5 * FN_2) + (1 * FP_2)
    
    return social_cost, FN_2, FP_2

def run_custom_grid_search():
    
    print("\n" + "=" * 75)
    print("  MLP HYPERPARAMETER TUNING (Custom Grid Search)")
    print("  Objective: Minimize Social Cost = 5*FN + 1*FP")
    print("=" * 75)
    
    print("\n[STEP 1] Loading Preprocessed Data")
    print("-" * 60)
    
    with open('nn_preprocessed_data.pkl', 'rb') as f:
        data = pickle.load(f)
    
    X_train = data['X_train']
    y_train = data['y_train']
    X_val = data['X_val']
    y_val = data['y_val']
    
    if sp.issparse(X_train):
        print("   Converting sparse matrices to dense...")
        X_train = X_train.toarray()
        X_val = X_val.toarray()
    
    X_train = np.nan_to_num(X_train, nan=0.0)
    X_val = np.nan_to_num(X_val, nan=0.0)
    
    print(f"   ✓ X_train: {X_train.shape}")
    print(f"   ✓ X_val: {X_val.shape}")
    
    print("\n[STEP 2] Defining Search Space")
    print("-" * 60)
    
    param_grid = list(product(ALPHA_VALUES, LEARNING_RATE_VALUES))
    total_combinations = len(param_grid)
    
    print(f"   Architecture: {HIDDEN_LAYER_SIZES} (Fixed)")
    print(f"   Alpha values: {ALPHA_VALUES}")
    print(f"   Learning rate values: {LEARNING_RATE_VALUES}")
    print(f"   Total combinations: {total_combinations}")
    
    print("\n[STEP 3] Running Custom Grid Search")
    print("-" * 60)
    print("\n   " + "=" * 70)
    print(f"   {'#':<3} {'Alpha':<10} {'LR':<10} {'Train Cost':<12} {'Val Cost':<12} {'Gap':<10} {'Time':<8}")
    print("   " + "=" * 70)
    
    results = []
    
    for i, (alpha, lr) in enumerate(param_grid, 1):
        start_time = time.time()
        
        model = MLPClassifier(
            hidden_layer_sizes=HIDDEN_LAYER_SIZES,
            activation='relu',
            solver='adam',
            alpha=alpha,
            learning_rate_init=lr,
            max_iter=MAX_ITER,
            early_stopping=EARLY_STOPPING,
            validation_fraction=VALIDATION_FRACTION,
            n_iter_no_change=N_ITER_NO_CHANGE,
            random_state=RANDOM_STATE,
            verbose=False
        )
        
        model.fit(X_train, y_train)
        
        y_train_pred = model.predict(X_train)
        train_cost, train_fn, train_fp = calculate_social_cost(y_train, y_train_pred)
        train_acc = accuracy_score(y_train, y_train_pred)
        
        y_val_pred = model.predict(X_val)
        val_cost, val_fn, val_fp = calculate_social_cost(y_val, y_val_pred)
        val_acc = accuracy_score(y_val, y_val_pred)
        
        gap = val_cost - train_cost
        
        elapsed = time.time() - start_time
        
        result = {
            'run': i,
            'alpha': alpha,
            'learning_rate_init': lr,
            'train_social_cost': train_cost,
            'train_fn_class2': train_fn,
            'train_fp_class2': train_fp,
            'train_accuracy': train_acc,
            'val_social_cost': val_cost,
            'val_fn_class2': val_fn,
            'val_fp_class2': val_fp,
            'val_accuracy': val_acc,
            'generalization_gap': gap,
            'iterations': model.n_iter_,
            'training_time_sec': elapsed
        }
        results.append(result)
        
        gap_str = f"+{gap}" if gap >= 0 else str(gap)
        print(f"   {i:<3} {alpha:<10} {lr:<10} {train_cost:<12,} {val_cost:<12,} {gap_str:<10} {elapsed:.1f}s")
    
    print("   " + "=" * 70)
    
    print("\n[STEP 4] Analyzing Results")
    print("-" * 60)
    
    results_df = pd.DataFrame(results)
    
    best_idx = results_df['val_social_cost'].idxmin()
    best_config = results_df.iloc[best_idx]
    
    print(f"\n   📊 RESULTS SUMMARY:")
    print(f"   " + "-" * 55)
    print(f"\n   Top 5 Configurations (by Val Social Cost):")
    top5 = results_df.nsmallest(5, 'val_social_cost')
    print(f"\n   {'Rank':<6} {'Alpha':<10} {'LR':<10} {'Val Cost':<12} {'Gap':<10}")
    print(f"   " + "-" * 45)
    for rank, (_, row) in enumerate(top5.iterrows(), 1):
        print(f"   {rank:<6} {row['alpha']:<10} {row['learning_rate_init']:<10} "
              f"{int(row['val_social_cost']):<12,} {int(row['generalization_gap']):<10}")
    
    print("\n\n" + "=" * 75)
    print("  🏆 BEST CONFIGURATION FOUND")
    print("=" * 75)
    
    results_df.to_csv("nn_tuning_results.csv", index=False)
    print(f"   Best alpha: {best_config['alpha']}")
    print(f"   Best learning rate: {best_config['learning_rate_init']}")
    print(f"   Best validation social cost: {int(best_config['val_social_cost']):,}")
    return results_df


if __name__ == "__main__":
    run_custom_grid_search()
