# Step 7: Social Cost Audit: model_audit.py

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    confusion_matrix, 
    precision_recall_fscore_support, 
    accuracy_score,
    ConfusionMatrixDisplay
)
from xgboost import XGBClassifier
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

BASE_PARAMS = {
    'n_estimators': 1000,
    'learning_rate': 0.05,
    'max_depth': 6,
    'objective': 'multi:softprob',
    'num_class': 3,
    'eval_metric': 'mlogloss',
    'use_label_encoder': False,
    'random_state': 42,
    'n_jobs': -1,
    'verbosity': 0,
    'early_stopping_rounds': 50
}

WEIGHTS_TO_AUDIT = [1, 5, 10, 20]

TR_01_RECALL_THRESHOLD = 0.50
TR_02_PRECISION_THRESHOLD = 0.80
TR_03_OVERFIT_THRESHOLD = 0.10
TR_04_EXPECTED_PROBA_COLUMNS = 3
TR_04_PROBABILITY_TOLERANCE = 1e-5
TR_05_LIFT_THRESHOLD = 3.0

CLASS_LABELS = ['Functional', 'Non-Functional', 'Needs Repair']

def calculate_social_cost(cm):
    FN_2 = cm[2, :].sum() - cm[2, 2]
    
    FP_2 = cm[:, 2].sum() - cm[2, 2]
    
    social_cost = (5 * FN_2) + (1 * FP_2)
    
    return social_cost, FN_2, FP_2

def validate_probability_output(proba, expected_rows):
    shape_ok = proba.shape == (expected_rows, TR_04_EXPECTED_PROBA_COLUMNS)
    finite_ok = np.isfinite(proba).all()
    range_ok = (proba >= -TR_04_PROBABILITY_TOLERANCE).all() and (
        proba <= 1 + TR_04_PROBABILITY_TOLERANCE
    ).all()

    row_sums = proba.sum(axis=1) if proba.ndim == 2 else np.array([])
    max_sum_error = (
        float(np.max(np.abs(row_sums - 1.0))) if len(row_sums) > 0 else float('inf')
    )
    row_sums_ok = np.allclose(
        row_sums, 1.0, atol=TR_04_PROBABILITY_TOLERANCE
    )

    return shape_ok, finite_ok, range_ok, row_sums_ok, max_sum_error

def save_confusion_matrix_heatmap(cm, weight, social_cost, filename):
    fig, ax = plt.subplots(figsize=(8, 6))
    
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=CLASS_LABELS, yticklabels=CLASS_LABELS,
                ax=ax, annot_kws={'size': 14})
    
    ax.set_xlabel('Predicted', fontsize=12)
    ax.set_ylabel('Actual', fontsize=12)
    ax.set_title(f'Confusion Matrix - Weight {weight}x\nSocial Cost Score: {social_cost:,}', 
                 fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()

def create_summary_dashboard(results_df):
    fig, ax1 = plt.subplots(figsize=(12, 7))
    
    weights = results_df['Weight']
    social_costs = results_df['Social_Cost']
    recall_c2 = results_df['Recall_C2'] * 100
    precision_c1 = results_df['Precision_C1'] * 100
    
    bar_width = 0.6
    bars = ax1.bar(weights, social_costs, bar_width, color='#3498db', alpha=0.7, 
                   label='Social Cost (Lower = Better)', edgecolor='black')
    
    ax1.set_xlabel('Class 2 Weight', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Social Cost Score', fontsize=12, color='#3498db')
    ax1.tick_params(axis='y', labelcolor='#3498db')
    ax1.set_xticks(weights)
    ax1.set_xticklabels([f'{w}x' for w in weights])
    
    for bar, cost in zip(bars, social_costs):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 50, 
                f'{int(cost):,}', ha='center', va='bottom', fontweight='bold', fontsize=11)
    
    ax2 = ax1.twinx()
    
    line1, = ax2.plot(weights, recall_c2, color='#27ae60', marker='o', linewidth=2.5, 
                      markersize=10, label='Class 2 Recall (%)')
    line2, = ax2.plot(weights, precision_c1, color='#e74c3c', marker='s', linewidth=2.5,
                      markersize=10, linestyle='--', label='Class 1 Precision (%)')
    
    ax2.set_ylabel('Recall / Precision (%)', fontsize=12)
    ax2.set_ylim(0, 105)
    
    for w, r, p in zip(weights, recall_c2, precision_c1):
        ax2.annotate(f'{r:.1f}%', (w, r), textcoords="offset points", 
                    xytext=(10, 5), fontsize=9, color='#27ae60', fontweight='bold')
        ax2.annotate(f'{p:.1f}%', (w, p), textcoords="offset points", 
                    xytext=(10, -10), fontsize=9, color='#e74c3c', fontweight='bold')
    
    plt.title('Model Audit Summary: Social Cost vs Performance Trade-off', 
              fontsize=14, fontweight='bold', pad=15)
    
    lines = [bars, line1, line2]
    labels = ['Social Cost', 'Class 2 Recall', 'Class 1 Precision']
    ax1.legend(lines, labels, loc='upper right', fontsize=10)
    
    ax1.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig('audit_tradeoff_summary.png', dpi=150, bbox_inches='tight')
    plt.close()

def run_model_audit():
    
    print("\n" + "=" * 75)
    print("  QUANTITATIVE MODEL AUDIT")
    print("  Tanzania Water Pump Challenge - Social Cost Optimization")
    print("=" * 75)
    
    print("\n[STEP 1] Loading Data")
    print("-" * 60)
    
    X = pd.read_parquet('final_X_train.parquet')
    y = pd.read_csv('final_y_train.csv')['status_group']
    
    object_cols = X.select_dtypes(include=['object']).columns.tolist()
    if len(object_cols) > 0:
        X = X.drop(columns=object_cols)
    X = X.fillna(0)
    
    print(f"   ✓ Features: {X.shape}")
    print(f"   ✓ Labels: {y.shape}")
    
    baseline_rate = (y == 2).mean()
    print(f"   ✓ Baseline Class 2 Rate: {baseline_rate:.4f} ({baseline_rate*100:.2f}%)")
    
    print("\n[STEP 2] Stratified Train/Val Split (80/20)")
    print("-" * 60)
    
    X_train, X_val, y_train, y_val = train_test_split(
        X, y,
        test_size=0.20,
        stratify=y,
        random_state=42
    )
    
    print(f"   ✓ Training: {len(X_train):,} samples")
    print(f"   ✓ Validation: {len(X_val):,} samples")
    
    print("\n[STEP 3] Running Model Audit Loop")
    print("-" * 60)
    print(f"   Auditing weights: {WEIGHTS_TO_AUDIT}")
    print(f"   Base Config: n_estimators={BASE_PARAMS['n_estimators']}, "
          f"lr={BASE_PARAMS['learning_rate']}, max_depth={BASE_PARAMS['max_depth']}")
    print()
    
    audit_results = []
    importance_model = None

    for W in WEIGHTS_TO_AUDIT:
        print(f"\n   {'='*55}")
        print(f"   AUDITING: Weight = {W}x")
        print(f"   {'='*55}")
        
        class_weights = {0: 1.0, 1: 1.0, 2: float(W)}
        sample_weights = y_train.map(class_weights).values
        
        model = XGBClassifier(**BASE_PARAMS)
        
        print(f"   Training...", end=" ", flush=True)
        train_start = time.time()
        model.fit(
            X_train, y_train,
            sample_weight=sample_weights,
            eval_set=[(X_val, y_val)],
            verbose=False
        )
        train_time = time.time() - train_start
        print(f"Done ({train_time:.1f}s)")

        if W == 5:
            importance_model = model
        
        y_train_pred = model.predict(X_train)
        train_acc = accuracy_score(y_train, y_train_pred)
        
        infer_start = time.time()
        y_val_proba = model.predict_proba(X_val)
        infer_time = time.time() - infer_start
        y_val_pred = model.classes_[np.argmax(y_val_proba, axis=1)]
         
        infer_per_10k = (infer_time / len(X_val)) * 10000
        (
            proba_shape_ok,
            proba_finite_ok,
            proba_range_ok,
            proba_rows_sum_ok,
            max_proba_sum_error,
        ) = validate_probability_output(y_val_proba, len(X_val))
        
        cm = confusion_matrix(y_val, y_val_pred, labels=[0, 1, 2])
        
        social_cost, FN_2, FP_2 = calculate_social_cost(cm)
        
        print(f"   📊 MATH VERIFICATION:")
        print(f"      FN_class2 (Missed Repairs): {FN_2}")
        print(f"      FP_class2 (False Alarms):   {FP_2}")
        print(f"      Social_Cost = (5 * {FN_2}) + (1 * {FP_2}) = {social_cost}")
        
        precision, recall, f1, _ = precision_recall_fscore_support(y_val, y_val_pred)
        val_acc = accuracy_score(y_val, y_val_pred)
        
        recall_c2 = recall[2]
        precision_c1 = precision[1]
        overfit_gap = abs(train_acc - val_acc)
        lift = recall_c2 / baseline_rate
        
        tr01_pass = recall_c2 > TR_01_RECALL_THRESHOLD
        tr02_pass = precision_c1 > TR_02_PRECISION_THRESHOLD
        tr03_pass = overfit_gap < TR_03_OVERFIT_THRESHOLD
        tr04_pass = (
            proba_shape_ok
            and proba_finite_ok
            and proba_range_ok
            and proba_rows_sum_ok
        )
        tr05_pass = lift > TR_05_LIFT_THRESHOLD
        
        all_pass = tr01_pass and tr02_pass and tr03_pass and tr04_pass and tr05_pass
        
        print(f"\n   📋 TECHNICAL COMPLIANCE:")
        print(f"      TR-01 (Recall > 0.50):     {recall_c2:.4f} {'✅ PASS' if tr01_pass else '❌ FAIL'}")
        print(f"      TR-02 (Precision > 0.80):  {precision_c1:.4f} {'✅ PASS' if tr02_pass else '❌ FAIL'}")
        print(f"      TR-03 (Overfit < 0.10):    {overfit_gap:.4f} {'✅ PASS' if tr03_pass else '❌ FAIL'}")
        print(f"      TR-04 (3 proba cols):      {y_val_proba.shape} {'✅ PASS' if proba_shape_ok else '❌ FAIL'}")
        print(f"      TR-04 (rows sum ≈ 1):      max error={max_proba_sum_error:.2e} {'✅ PASS' if proba_rows_sum_ok else '❌ FAIL'}")
        print(f"      TR-05 (Lift > 3.0):        {lift:.4f} {'✅ PASS' if tr05_pass else '❌ FAIL'}")
        print(f"\n      OVERALL: {'✅ ALL PASS' if all_pass else '❌ SOME FAILED'}")
        
        cm_filename = f'cm_weight_{W}.png'
        save_confusion_matrix_heatmap(cm, W, social_cost, cm_filename)
        print(f"\n   ✓ Saved: {cm_filename}")
        
        audit_results.append({
            'Weight': W,
            'Social_Cost': social_cost,
            'FN_Class2': FN_2,
            'FP_Class2': FP_2,
            'Recall_C2': recall_c2,
            'Precision_C1': precision_c1,
            'Accuracy': val_acc,
            'Train_Acc': train_acc,
            'Overfit_Gap': overfit_gap,
            'Lift': lift,
            'Predict_Proba_10k': infer_per_10k,
            'Proba_Shape_OK': proba_shape_ok,
            'Proba_Finite_OK': proba_finite_ok,
            'Proba_Range_OK': proba_range_ok,
            'Proba_Rows_Sum_OK': proba_rows_sum_ok,
            'Proba_Max_Row_Sum_Error': max_proba_sum_error,
            'TR01_PASS': tr01_pass,
            'TR02_PASS': tr02_pass,
            'TR03_PASS': tr03_pass,
            'TR04_PASS': tr04_pass,
            'TR05_PASS': tr05_pass,
            'ALL_PASS': all_pass
        })
    
    print("\n\n[STEP 4] Generating Summary Dashboard")
    print("-" * 60)
    
    results_df = pd.DataFrame(audit_results)
    create_summary_dashboard(results_df)
    print("   ✓ Saved: audit_tradeoff_summary.png")
    
    results_df.to_csv('audit_results.csv', index=False)
    print("   ✓ Saved: audit_results.csv")

    if importance_model is not None:
        print("\n\n[STEP 5] XGBoost Feature Importance (Weight 5x model)")
        print("-" * 60)
        importance_df = pd.DataFrame({
            'feature': X_train.columns,
            'importance': importance_model.feature_importances_
        }).sort_values('importance', ascending=False)

        print("   Top 10 features by importance:")
        for i, (_, row) in enumerate(importance_df.head(10).iterrows(), 1):
            print(f"      {i:2d}. {row['feature']:<40} {row['importance']:.4f}")

        importance_df.to_csv('feature_importance.csv', index=False)
        print("   ✓ Saved: feature_importance.csv")

        fig, ax = plt.subplots(figsize=(10, 8))
        top20 = importance_df.head(20).iloc[::-1]
        ax.barh(top20['feature'], top20['importance'], color='steelblue')
        ax.set_xlabel('Feature Importance')
        ax.set_title('Top 20 Most Important Features - XGBoost (Weight 5x)')
        plt.tight_layout()
        plt.savefig('feature_importance.png', dpi=150, bbox_inches='tight')
        plt.close()
        print("   ✓ Saved: feature_importance.png")

    print("\n\n" + "=" * 75)
    print("  FINAL AUDIT REPORT")
    print("=" * 75)
    
    print("\n   📊 SUMMARY TABLE:")
    print("   " + "-" * 70)
    print(f"   {'Weight':<8} {'Social_Cost':>12} {'Recall_C2':>12} {'Prec_C1':>12} {'Lift':>10} {'Status':>10}")
    print("   " + "-" * 70)
    
    for _, row in results_df.iterrows():
        status = "✅ PASS" if row['ALL_PASS'] else "❌ FAIL"
        print(f"   {int(row['Weight']):<8} {int(row['Social_Cost']):>12,} "
              f"{row['Recall_C2']:>12.4f} {row['Precision_C1']:>12.4f} "
              f"{row['Lift']:>10.2f} {status:>10}")
    
    print("   " + "-" * 70)
    
    passing_results = results_df[results_df['ALL_PASS']]
    if len(passing_results) > 0:
        optimal = passing_results.loc[passing_results['Social_Cost'].idxmin()]
        print(f"\n   🏆 OPTIMAL CONFIGURATION:")
        print(f"      Weight: {int(optimal['Weight'])}x")
        print(f"      Social Cost: {int(optimal['Social_Cost']):,}")
        print(f"      Recall (Class 2): {optimal['Recall_C2']:.4f} ({optimal['Recall_C2']*100:.1f}%)")
        print(f"      Precision (Class 1): {optimal['Precision_C1']:.4f} ({optimal['Precision_C1']*100:.1f}%)")
    else:
        print("\n   ⚠️  No weight configuration passed all TR thresholds.")
        best = results_df.loc[results_df['Social_Cost'].idxmin()]
        print(f"   Best Social Cost: Weight {int(best['Weight'])}x = {int(best['Social_Cost']):,}")
    
    print("\n" + "=" * 75)
    print("  AUDIT COMPLETE")
    print("=" * 75 + "\n")
    
    return results_df

if __name__ == "__main__":
    
    print("Starting Quantitative Model Audit...")
    print("(This will train 4 models with full fidelity - ~10 minutes)\n")
    
    try:
        results = run_model_audit()
        print("Audit complete! All reports and visualizations saved.")
        
    except FileNotFoundError as e:
        print(f"Error: Could not find file - {e}")
        print("Make sure you have run feature_engineering.py first.")
    except ImportError as e:
        print(f"Error: Missing package - {e}")
        print("Install with: pip install xgboost scikit-learn seaborn")
