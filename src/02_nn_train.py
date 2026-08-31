# Step 9: MLP Training & Audit: 02_nn_train.py

import numpy as np
import pandas as pd
import pickle
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    accuracy_score,
    precision_recall_fscore_support
)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import io
import time
import warnings

warnings.filterwarnings('ignore')

if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

HIDDEN_LAYER_SIZES = (128, 64, 32)

ACTIVATION = 'relu'

SOLVER = 'adam'

ALPHA = 0.05

MAX_ITER = 500
EARLY_STOPPING = True
VALIDATION_FRACTION = 0.1
N_ITER_NO_CHANGE = 20
RANDOM_STATE = 42

CLASS_LABELS = ['Functional', 'Non-Functional', 'Needs Repair']

def calculate_social_cost(cm):
    FN_2 = cm[2, :].sum() - cm[2, 2]
    
    FP_2 = cm[:, 2].sum() - cm[2, 2]
    
    social_cost = (5 * FN_2) + (1 * FP_2)
    
    return social_cost, FN_2, FP_2

def save_confusion_matrix_heatmap(cm, social_cost, filename):
    fig, ax = plt.subplots(figsize=(8, 6))
    
    sns.heatmap(cm, annot=True, fmt='d', cmap='Purples', 
                xticklabels=CLASS_LABELS, yticklabels=CLASS_LABELS,
                ax=ax, annot_kws={'size': 14})
    
    ax.set_xlabel('Predicted', fontsize=12)
    ax.set_ylabel('Actual', fontsize=12)
    ax.set_title(f'MLP Confusion Matrix\nSocial Cost Score: {social_cost:,}', 
                 fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()

def plot_training_curve(loss_curve, filename):
    fig, ax = plt.subplots(figsize=(10, 5))
    
    ax.plot(loss_curve, color='#9b59b6', linewidth=2)
    ax.set_xlabel('Iteration', fontsize=12)
    ax.set_ylabel('Loss', fontsize=12)
    ax.set_title('MLP Training Loss Curve', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()

def train_mlp_challenger():
    
    print("\n" + "=" * 70)
    print("  MLP CHALLENGER MODEL - Neural Network Training")
    print("  Tanzania Water Pump Challenge")
    print("=" * 70)
    
    print("\n[STEP 1] Loading Preprocessed Data")
    print("-" * 55)
    
    with open('nn_preprocessed_data.pkl', 'rb') as f:
        data = pickle.load(f)
    
    X_train = data['X_train']
    y_train = data['y_train']
    X_val = data['X_val']
    y_val = data['y_val']
    config = data['config']
    
    print(f"   ✓ X_train shape: {X_train.shape}")
    print(f"   ✓ y_train shape: {y_train.shape}")
    print(f"   ✓ X_val shape: {X_val.shape}")
    print(f"   ✓ y_val shape: {y_val.shape}")
    print(f"   ✓ Preprocessing config: {config}")
    
    import scipy.sparse as sp
    if sp.issparse(X_train):
        print("\n   Converting sparse matrices to dense...")
        X_train = X_train.toarray()
        X_val = X_val.toarray()
        print(f"   ✓ Converted to dense arrays")
    
    if np.any(np.isnan(X_train)):
        print("   ⚠️  Found NaN values, replacing with 0...")
        X_train = np.nan_to_num(X_train, nan=0.0)
        X_val = np.nan_to_num(X_val, nan=0.0)
        print(f"   ✓ NaN values handled")
    
    print(f"\n   Training Class Distribution:")
    for c in [0, 1, 2]:
        count = (y_train == c).sum()
        pct = count / len(y_train) * 100
        print(f"      Class {c}: {count:,} ({pct:.2f}%)")
    
    print("\n[STEP 2] Initializing MLP Classifier")
    print("-" * 55)
    
    model = MLPClassifier(
        hidden_layer_sizes=HIDDEN_LAYER_SIZES,
        activation=ACTIVATION,
        solver=SOLVER,
        alpha=ALPHA,
        max_iter=MAX_ITER,
        early_stopping=EARLY_STOPPING,
        validation_fraction=VALIDATION_FRACTION,
        n_iter_no_change=N_ITER_NO_CHANGE,
        random_state=RANDOM_STATE,
        verbose=True
    )
    
    print(f"   Architecture:")
    print(f"      Input Layer:  {X_train.shape[1]} neurons (auto-inferred)")
    print(f"      Hidden Layer 1: {HIDDEN_LAYER_SIZES[0]} neurons (ReLU)")
    print(f"      Hidden Layer 2: {HIDDEN_LAYER_SIZES[1]} neurons (ReLU)")
    print(f"      Hidden Layer 3: {HIDDEN_LAYER_SIZES[2]} neurons (ReLU)")
    print(f"      Output Layer: 3 neurons (Softmax implicit)")
    print(f"\n   Optimization:")
    print(f"      Solver: {SOLVER}")
    print(f"      L2 Alpha: {ALPHA}")
    print(f"      Early Stopping: {EARLY_STOPPING}")
    print(f"      Max Iterations: {MAX_ITER}")
    
    print("\n[STEP 3] Training MLP Model")
    print("-" * 55)
    print("   Training with early stopping...")
    print("   (Will stop if no improvement for 20 iterations)\n")
    
    train_start = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - train_start
    
    print(f"\n   ✓ Training complete!")
    print(f"   ✓ Training time: {train_time:.2f} seconds")
    print(f"   ✓ Iterations completed: {model.n_iter_}")
    print(f"   ✓ Final loss: {model.loss_:.4f}")
    
    print("\n[STEP 4] Evaluating on Validation Set")
    print("-" * 55)
    
    y_pred = model.predict(X_val)
    y_pred_proba = model.predict_proba(X_val)
    
    cm = confusion_matrix(y_val, y_pred, labels=[0, 1, 2])
    
    precision, recall, f1, support = precision_recall_fscore_support(y_val, y_pred)
    accuracy = accuracy_score(y_val, y_pred)
    
    social_cost, FN_2, FP_2 = calculate_social_cost(cm)
    
    print(f"\n   📊 CONFUSION MATRIX:")
    print(f"   " + "-" * 50)
    print(f"\n   Predicted →")
    print(f"   Actual ↓    ", end="")
    for label in CLASS_LABELS:
        print(f"{label[:12]:>12}", end=" ")
    print()
    
    for i, (label, row) in enumerate(zip(CLASS_LABELS, cm)):
        print(f"   {label[:12]:<12}", end=" ")
        for val in row:
            print(f"{val:>12,}", end=" ")
        print()
    
    print("\n\n" + "=" * 70)
    print("  💰 SOCIAL COST AUDIT")
    print("=" * 70)
    
    print(f"\n   📊 MATH VERIFICATION:")
    print(f"      FN_class2 (Missed Repairs): {FN_2}")
    print(f"      FP_class2 (False Alarms):   {FP_2}")
    print(f"\n      Social_Cost = (5 × {FN_2}) + (1 × {FP_2})")
    print(f"                  = {5 * FN_2} + {FP_2}")
    
    print("\n   " + "=" * 50)
    print(f"   🎯 TOTAL SOCIAL COST: {social_cost:,}")
    print("   " + "=" * 50)
    
    print("\n\n[STEP 6] Classification Report")
    print("-" * 55)
    
    report = classification_report(y_val, y_pred, target_names=CLASS_LABELS, digits=4)
    print(report)
    
    print(f"\n   Per-Class Performance Summary:")
    print(f"   " + "-" * 50)
    print(f"   {'Class':<20} {'Precision':>10} {'Recall':>10} {'F1':>10}")
    print(f"   " + "-" * 50)
    
    for i, label in enumerate(CLASS_LABELS):
        print(f"   {label:<20} {precision[i]:>10.4f} {recall[i]:>10.4f} {f1[i]:>10.4f}")
    
    print(f"\n   Overall Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    
    baseline_rate = (y_val == 2).mean()
    lift = recall[2] / baseline_rate if baseline_rate > 0 else 0
    
    print(f"   Class 2 Recall: {recall[2]:.4f} ({recall[2]*100:.2f}%)")
    print(f"   Class 2 Lift: {lift:.2f}x (vs baseline {baseline_rate*100:.2f}%)")

    print("\n   [REPORT NUMBERS] Single-run values (confusion matrix and metrics are consistent):")
    for label, row in zip(CLASS_LABELS, cm):
        print(f"      {label:<16}: {row.tolist()}")
    print(f"      Class 2: TP={cm[2, 2]}, FN={FN_2}, FP={FP_2}")
    print(f"      Social Cost = 5*{FN_2} + {FP_2} = {social_cost}")
    print(f"      Recall(C2)={recall[2]:.4f}  Precision(C1)={precision[1]:.4f}  "
          f"Accuracy={accuracy:.4f}  Lift={lift:.2f}x")

    print("\n[STEP 7] Saving Audit Results")
    print("-" * 55)
    
    save_confusion_matrix_heatmap(cm, social_cost, 'nn_confusion_matrix.png')
    print("   ✓ Saved: nn_confusion_matrix.png")
    
    if hasattr(model, 'loss_curve_'):
        plot_training_curve(model.loss_curve_, 'nn_training_curve.png')
        print("   ✓ Saved: nn_training_curve.png")
    
    audit_results = pd.DataFrame([{
        'Model': 'MLP_Challenger',
        'Architecture': '128-64-32',
        'Activation': ACTIVATION,
        'Alpha': ALPHA,
        'Iterations': model.n_iter_,
        'Training_Time_Sec': train_time,
        'Social_Cost': social_cost,
        'FN_Class2': FN_2,
        'FP_Class2': FP_2,
        'Accuracy': accuracy,
        'Class0_Recall': recall[0],
        'Class1_Recall': recall[1],
        'Class2_Recall': recall[2],
        'Class0_Precision': precision[0],
        'Class1_Precision': precision[1],
        'Class2_Precision': precision[2],
        'Lift_Class2': lift
    }])
    
    audit_results.to_csv('nn_audit_results.csv', index=False)
    print("   ✓ Saved: nn_audit_results.csv")
    
    with open('nn_model.pkl', 'wb') as f:
        pickle.dump(model, f)
    print("   ✓ Saved: nn_model.pkl")
    
    print("\n\n" + "=" * 70)
    print("  MLP CHALLENGER MODEL - TRAINING COMPLETE")
    print("=" * 70)
    print(f"   Overall Accuracy: {accuracy:.4f}")
    print(f"   Class 2 Recall: {recall[2]:.4f}")
    print(f"   Class 2 Lift: {lift:.2f}x")
    print(f"   Total Social Cost: {social_cost:,}")
    return model, audit_results


if __name__ == "__main__":
    train_mlp_challenger()
