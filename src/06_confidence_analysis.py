# Step 12: TR-04 Confidence Analysis: 06_confidence_analysis.py

import pandas as pd
import numpy as np
import pickle
import sys
import io
import warnings

warnings.filterwarnings('ignore')

if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

CLASS_LABELS = {0: 'Functional', 1: 'Non-Functional', 2: 'Needs Repair'}
VAL_SIZE = 0.20
RANDOM_STATE = 42
CLASS_WEIGHTS = {0: 1.0, 1: 1.0, 2: 5.0}
EXPECTED_PROBA_COLUMNS = 3
PROBABILITY_TOLERANCE = 1e-5

def validate_probability_output(model_name, proba):
    shape_ok = proba.shape[1] == EXPECTED_PROBA_COLUMNS if proba.ndim == 2 else False
    finite_ok = np.isfinite(proba).all()
    range_ok = (proba >= -PROBABILITY_TOLERANCE).all() and (
        proba <= 1 + PROBABILITY_TOLERANCE
    ).all()
    row_sums = proba.sum(axis=1) if proba.ndim == 2 else np.array([])
    max_sum_error = (
        float(np.max(np.abs(row_sums - 1.0))) if len(row_sums) > 0 else float('inf')
    )
    row_sums_ok = np.allclose(row_sums, 1.0, atol=PROBABILITY_TOLERANCE)
    passed = shape_ok and finite_ok and range_ok and row_sums_ok

    print(f"   {model_name} probability shape: {proba.shape}")
    print(f"   {model_name} has 3 probability columns: {'PASS' if shape_ok else 'FAIL'}")
    print(f"   {model_name} probability rows sum to 1: max error={max_sum_error:.2e} {'PASS' if row_sums_ok else 'FAIL'}")

    if not passed:
        raise ValueError(f"{model_name} failed TR-04 probability-output validation.")

    return max_sum_error

print("\n" + "=" * 70)
print("  TR-04 PROBABILISTIC CONFIDENCE ANALYSIS")
print("=" * 70)

print("\n[STEP 1] Training XGBoost (W=5x) for predict_proba()")
print("-" * 55)
sys.stdout.flush()

from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

X_full = pd.read_parquet('final_X_train.parquet')
y_full = pd.read_csv('final_y_train.csv')['status_group']

object_cols = X_full.select_dtypes(include=['object']).columns.tolist()
X_full = X_full.drop(columns=object_cols)
X_full = X_full.fillna(0)

X_train_xgb, X_val_xgb, y_train_xgb, y_val_xgb = train_test_split(
    X_full, y_full, test_size=VAL_SIZE, stratify=y_full, random_state=RANDOM_STATE
)

sample_weights = y_train_xgb.map(CLASS_WEIGHTS).values

xgb_model = XGBClassifier(
    n_estimators=1000, learning_rate=0.05, max_depth=6,
    objective='multi:softprob', num_class=3, eval_metric='mlogloss',
    use_label_encoder=False, random_state=RANDOM_STATE, n_jobs=-1,
    verbosity=0, early_stopping_rounds=50
)
xgb_model.fit(X_train_xgb, y_train_xgb, sample_weight=sample_weights,
              eval_set=[(X_val_xgb, y_val_xgb)], verbose=False)

xgb_proba = xgb_model.predict_proba(X_val_xgb)
xgb_pred = xgb_model.predict(X_val_xgb)
xgb_max_sum_error = validate_probability_output("XGBoost", xgb_proba)

print(f"   XGBoost predict_proba() shape: {xgb_proba.shape}")
print(f"   Columns: P(Class 0), P(Class 1), P(Class 2)")
print(f"   Sum of probabilities per sample (first 5): {xgb_proba[:5].sum(axis=1)}")
sys.stdout.flush()

print("\n[STEP 2] Loading MLP model for predict_proba()")
print("-" * 55)
sys.stdout.flush()

with open('nn_model.pkl', 'rb') as f:
    mlp_model = pickle.load(f)

with open('nn_preprocessed_data.pkl', 'rb') as f:
    nn_data = pickle.load(f)

X_val_mlp = nn_data['X_val']
y_val_mlp = nn_data['y_val']

import scipy.sparse as sp
if sp.issparse(X_val_mlp):
    X_val_mlp = X_val_mlp.toarray()
X_val_mlp = np.nan_to_num(X_val_mlp, nan=0.0)
print(f"   Converted to dense and cleaned NaN: shape {X_val_mlp.shape}")

mlp_proba = mlp_model.predict_proba(X_val_mlp)
mlp_pred = mlp_model.predict(X_val_mlp)
mlp_max_sum_error = validate_probability_output("MLP", mlp_proba)

print(f"   MLP predict_proba() shape: {mlp_proba.shape}")
print(f"   Columns: P(Class 0), P(Class 1), P(Class 2)")
print(f"   Sum of probabilities per sample (first 5): {mlp_proba[:5].sum(axis=1)}")
sys.stdout.flush()

print("\n[STEP 3] Generating Sample Prediction Table")
print("-" * 55)
sys.stdout.flush()

y_val_arr = y_val_xgb.values

np.random.seed(42)

correct_high_conf = np.where((xgb_pred == y_val_arr) & (xgb_proba.max(axis=1) > 0.8))[0]
correct_low_conf = np.where((xgb_pred == y_val_arr) & (xgb_proba.max(axis=1) < 0.6))[0]
incorrect = np.where(xgb_pred != y_val_arr)[0]
class2_correct = np.where((xgb_pred == 2) & (y_val_arr == 2))[0]
class2_missed = np.where((xgb_pred != 2) & (y_val_arr == 2))[0]

sample_indices = []

if len(correct_high_conf) >= 2:
    candidates = correct_high_conf
    c0 = candidates[y_val_arr[candidates] == 0]
    c1 = candidates[y_val_arr[candidates] == 1]
    if len(c0) > 0: sample_indices.append(np.random.choice(c0))
    if len(c1) > 0: sample_indices.append(np.random.choice(c1))

if len(correct_low_conf) > 0:
    sample_indices.append(np.random.choice(correct_low_conf))

if len(class2_correct) >= 2:
    sample_indices.extend(np.random.choice(class2_correct, min(2, len(class2_correct)), replace=False))

if len(class2_missed) > 0:
    sample_indices.append(np.random.choice(class2_missed))

wrong_not_c2 = np.where((xgb_pred != y_val_arr) & (y_val_arr != 2))[0]
if len(wrong_not_c2) >= 2:
    sample_indices.extend(np.random.choice(wrong_not_c2, 2, replace=False))

sample_indices = list(set(sample_indices))[:8]

samples = []
for idx in sample_indices:
    samples.append({
        'Sample': f"Val#{idx}",
        'P(Functional)': f"{xgb_proba[idx, 0]:.3f}",
        'P(Non-Func)': f"{xgb_proba[idx, 1]:.3f}",
        'P(Repair)': f"{xgb_proba[idx, 2]:.3f}",
        'Predicted': CLASS_LABELS[xgb_pred[idx]],
        'Actual': CLASS_LABELS[y_val_arr[idx]],
        'Correct': 'Yes' if xgb_pred[idx] == y_val_arr[idx] else 'No',
        'Max Conf': f"{xgb_proba[idx].max():.3f}"
    })

sample_df = pd.DataFrame(samples)
sample_df.to_csv('confidence_samples.csv', index=False)
print(f"   Saved: confidence_samples.csv ({len(samples)} samples)")

print("\n   Markdown table for report:\n")
print("| Sample | P(Functional) | P(Non-Func) | P(Repair) | Predicted | Actual | Correct | Max Conf |")
print("|--------|:---:|:---:|:---:|-----------|--------|:---:|:---:|")
for s in samples:
    print(f"| {s['Sample']} | {s['P(Functional)']} | {s['P(Non-Func)']} | {s['P(Repair)']} | {s['Predicted']} | {s['Actual']} | {s['Correct']} | {s['Max Conf']} |")

sys.stdout.flush()

print("\n\n[STEP 4] Confidence Statistics")
print("-" * 55)
sys.stdout.flush()

for name, proba, pred, y_true in [
    ("XGBoost (W=5x)", xgb_proba, xgb_pred, y_val_arr),
    ("MLP Baseline", mlp_proba, mlp_pred, y_val_mlp)
]:
    max_conf = proba.max(axis=1)
    correct_mask = (pred == y_true)
    
    print(f"\n   {name}:")
    print(f"      Mean max confidence:          {max_conf.mean():.4f}")
    print(f"      Mean confidence (correct):    {max_conf[correct_mask].mean():.4f}")
    print(f"      Mean confidence (incorrect):  {max_conf[~correct_mask].mean():.4f}")
    print(f"      P(Class 2) range:             [{proba[:, 2].min():.4f}, {proba[:, 2].max():.4f}]")
    print(f"      P(Class 2) mean:              {proba[:, 2].mean():.4f}")
    print(f"      Samples with P(Class 2)>0.5:  {(proba[:, 2] > 0.5).sum()}")
    
sys.stdout.flush()

print("\n[STEP 5] Generating P(Class 2) Distribution Histogram")
print("-" * 55)
sys.stdout.flush()

fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)

colors = {'all': '#4A90D9', 'actual_c2': '#E74C3C', 'threshold': '#2ECC71'}

for ax, name, proba, y_true in [
    (axes[0], "XGBoost (W=5×)", xgb_proba, y_val_arr),
    (axes[1], "MLP Challenger (α=0.05)", mlp_proba, y_val_mlp)
]:
    p_class2 = proba[:, 2]
    actual_c2_mask = (y_true == 2)
    
    ax.hist(p_class2[~actual_c2_mask], bins=50, alpha=0.6, color=colors['all'],
            label=f'Actual Class 0/1 (n={( ~actual_c2_mask).sum():,})', edgecolor='white', linewidth=0.5)
    
    ax.hist(p_class2[actual_c2_mask], bins=50, alpha=0.8, color=colors['actual_c2'],
            label=f'Actual Class 2 (n={actual_c2_mask.sum():,})', edgecolor='white', linewidth=0.5)
    
    ax.axvline(x=0.5, color=colors['threshold'], linestyle='--', linewidth=1.5,
               label='Decision threshold (0.5)')
    
    ax.set_xlabel('P(Class 2 | x)', fontsize=12)
    ax.set_title(name, fontsize=13, fontweight='bold')
    ax.legend(fontsize=9, loc='upper right')
    ax.set_xlim(-0.02, 1.02)
    ax.grid(axis='y', alpha=0.3)

axes[0].set_ylabel('Number of Samples', fontsize=12)

fig.suptitle('TR-04 Evidence: Probabilistic Confidence Output P(y | x)\n'
             'Distribution of P(Class 2 = "Needs Repair" | x) across validation set',
             fontsize=14, fontweight='bold', y=1.02)

plt.tight_layout()
plt.savefig('confidence_distribution.png', dpi=150, bbox_inches='tight')
plt.close()
print("   Saved: confidence_distribution.png")

print("\n" + "=" * 70)
print("  TR-04 CONFIDENCE ANALYSIS COMPLETE")
print("=" * 70)
print("   Both models output per-class probabilities via predict_proba().")
print(f"   Probability row-sum max errors: XGBoost={xgb_max_sum_error:.2e}, MLP={mlp_max_sum_error:.2e}")
print("   Files saved: confidence_distribution.png and confidence_samples.csv")
print("   TR-04 verdict: PASS for both models.")
