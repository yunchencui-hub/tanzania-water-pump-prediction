# Step 3: Strategic EDA: strategic_eda.py

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple
import sys
import io
import os
import warnings

warnings.filterwarnings('ignore')

if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

sns.set(style="whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 10

CLASS_LABELS = {
    0: 'Functional',
    1: 'Non-Functional',
    2: 'Needs Repair'
}

SURE_KILL_FEATURES = ['quantity', 'extraction_type_class', 'payment_type', 'source_class']
HIDDEN_RISK_FEATURES = ['source_class', 'waterpoint_type', 'management', 'scheme_management']

HIGH_RISK_RATIO_THRESHOLD = 2.0
HOTSPOT_FAILURE_THRESHOLD = 0.60

REFERENCE_YEAR = 2014

def calculate_risk_ratio(
    df: pd.DataFrame,
    feature: str,
    target_class: int = 1,
    target_col: str = 'status_group'
) -> pd.DataFrame:
    p_global = (df[target_col] == target_class).mean()
    
    risk_stats = []
    
    for value in df[feature].unique():
        mask = df[feature] == value
        subset = df[mask]
        
        count = len(subset)
        target_count = (subset[target_col] == target_class).sum()
        p_conditional = target_count / count if count > 0 else 0
        
        risk_ratio = p_conditional / p_global if p_global > 0 else 0
        
        risk_stats.append({
            'Feature': feature,
            'Value': value,
            'Count': count,
            'Target_Count': target_count,
            'P_Conditional': round(p_conditional * 100, 2),
            'P_Global': round(p_global * 100, 2),
            'Risk_Ratio': round(risk_ratio, 3)
        })
    
    risk_df = pd.DataFrame(risk_stats)
    risk_df = risk_df.sort_values('Risk_Ratio', ascending=False)
    
    return risk_df

def analyze_sure_kill_signals(
    df: pd.DataFrame,
    features: List[str],
    target_col: str = 'status_group'
) -> Dict:
    print("\n" + "=" * 70)
    print("  ANALYSIS 1: 'Sure-Kill' Signal Hunt (Budget Optimization)")
    print("=" * 70)
    print("  Goal: Find feature values where Risk Ratio > 2.0")
    print("  These are candidates for Rule-Based Filtering\n")
    
    all_high_risk = []
    
    for feature in features:
        if feature not in df.columns:
            print(f"  ⚠️  Feature '{feature}' not found, skipping...")
            continue
            
        risk_df = calculate_risk_ratio(df, feature, target_class=1)
        
        high_risk = risk_df[risk_df['Risk_Ratio'] > HIGH_RISK_RATIO_THRESHOLD]
        
        print(f"\n  📊 Feature: {feature}")
        print(f"     High-Risk Values (RR > {HIGH_RISK_RATIO_THRESHOLD}):")
        
        if len(high_risk) > 0:
            for _, row in high_risk.iterrows():
                print(f"        → {row['Value']}: RR = {row['Risk_Ratio']:.2f}x "
                      f"({row['P_Conditional']:.1f}% vs {row['P_Global']:.1f}% global)")
                all_high_risk.append(row.to_dict())
        else:
            print(f"        No values exceed threshold")
    
    all_high_risk_df = pd.DataFrame(all_high_risk)
    if len(all_high_risk_df) > 0:
        all_high_risk_df = all_high_risk_df.sort_values('Risk_Ratio', ascending=False)
    
    return {
        'high_risk_rules': all_high_risk_df,
        'threshold': HIGH_RISK_RATIO_THRESHOLD
    }

def analyze_hidden_risk_signals(
    df: pd.DataFrame,
    features: List[str],
    target_col: str = 'status_group'
) -> Dict:
    print("\n" + "=" * 70)
    print("  ANALYSIS 2: 'Hidden Risk' Hunter (Social Cost Minimization)")
    print("=" * 70)
    print("  Goal: Distinguish 'Functional' (0) from 'Needs Repair' (2)")
    print("  Excluding 'Non-Functional' pumps from this analysis\n")
    
    df_binary = df[df[target_col].isin([0, 2])].copy()
    
    print(f"  📊 Binary Subset: {len(df_binary):,} samples")
    print(f"     Class 0 (Functional):    {(df_binary[target_col] == 0).sum():,}")
    print(f"     Class 2 (Needs Repair):  {(df_binary[target_col] == 2).sum():,}")
    
    discriminative_features = []
    
    for feature in features:
        if feature not in df_binary.columns:
            continue
        
        contingency = pd.crosstab(
            df_binary[feature], 
            df_binary[target_col], 
            normalize='columns'
        ) * 100
        
        if 0 in contingency.columns and 2 in contingency.columns:
            contingency['Diff'] = abs(contingency[0] - contingency[2])
            top_discriminative = contingency.nlargest(3, 'Diff')
            
            max_diff = contingency['Diff'].max()
            
            print(f"\n  📊 Feature: {feature}")
            print(f"     Max Class Difference: {max_diff:.1f}%")
            print(f"     Top Discriminative Values:")
            
            for value, row in top_discriminative.iterrows():
                print(f"        → {value}: Class0={row[0]:.1f}%, Class2={row[2]:.1f}% (Δ={row['Diff']:.1f}%)")
            
            discriminative_features.append({
                'Feature': feature,
                'Max_Diff': max_diff,
                'Top_Value': top_discriminative.index[0]
            })
    
    if len(discriminative_features) > 0:
        top_feature = max(discriminative_features, key=lambda x: x['Max_Diff'])['Feature']
        _plot_class_comparison(df_binary, top_feature, target_col)
    
    return {
        'discriminative_features': pd.DataFrame(discriminative_features),
        'subset_size': len(df_binary)
    }

def _plot_class_comparison(
    df: pd.DataFrame,
    feature: str,
    target_col: str = 'status_group'
) -> None:
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    contingency = pd.crosstab(
        df[feature], 
        df[target_col], 
        normalize='index'
    ) * 100
    
    contingency.columns = ['Functional', 'Needs Repair']
    
    contingency.plot(kind='bar', stacked=True, ax=ax, 
                     color=['#2ecc71', '#e74c3c'], edgecolor='white')
    
    ax.set_xlabel(feature.replace('_', ' ').title())
    ax.set_ylabel('Percentage (%)')
    ax.set_title(f"Class Distribution by {feature}: Functional vs Needs Repair")
    ax.legend(title='Status', loc='upper right')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
    
    ax.axhline(y=50, color='gray', linestyle='--', alpha=0.5, label='50% baseline')
    
    plt.tight_layout()
    plt.savefig('hidden_risk_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\n  ✓ Saved: hidden_risk_comparison.png")

def analyze_temporal_physical_signals(
    df: pd.DataFrame,
    target_col: str = 'status_group'
) -> Dict:
    print("\n" + "=" * 70)
    print("  ANALYSIS 3: Temporal & Physical Signal Analysis")
    print("=" * 70)
    
    results = {}
    
    print("\n  📊 PUMP AGE ANALYSIS:")
    print("-" * 50)
    
    if 'construction_year' in df.columns:
        df_age = df[df['construction_year'] > 0].copy()
        df_age['pump_age'] = REFERENCE_YEAR - df_age['construction_year']
        
        print(f"     Samples with valid construction_year: {len(df_age):,}")
        print(f"     Samples excluded (year == 0): {(df['construction_year'] == 0).sum():,}")
        
        df_age['age_bucket'] = pd.cut(
            df_age['pump_age'], 
            bins=[0, 5, 10, 15, 20, 30, 50, 100],
            labels=['0-5', '6-10', '11-15', '16-20', '21-30', '31-50', '51+']
        )
        
        age_failure = df_age.groupby('age_bucket', observed=True).apply(
            lambda x: pd.Series({
                'count': len(x),
                'failure_rate': (x[target_col] == 1).mean() * 100,
                'repair_rate': (x[target_col] == 2).mean() * 100
            })
        ).reset_index()
        
        print(f"\n     Failure Rate by Pump Age:")
        for _, row in age_failure.iterrows():
            print(f"        {row['age_bucket']} years: "
                  f"{row['failure_rate']:.1f}% failed, {row['repair_rate']:.1f}% needs repair "
                  f"(n={int(row['count']):,})")
        
        results['age_analysis'] = age_failure
        
        _plot_age_analysis(df_age, target_col)
    
    print("\n  📊 ZERO-VALUE RISK TEST:")
    print("-" * 50)
    print("     Hypothesis: Zero values signal higher failure risk\n")
    
    zero_risk_tests = []
    
    for col in ['amount_tsh', 'construction_year', 'population']:
        if col not in df.columns:
            continue
        
        zero_group = df[df[col] == 0]
        nonzero_group = df[df[col] > 0]
        
        zero_failure = (zero_group[target_col] == 1).mean() * 100
        nonzero_failure = (nonzero_group[target_col] == 1).mean() * 100
        
        zero_repair = (zero_group[target_col] == 2).mean() * 100
        nonzero_repair = (nonzero_group[target_col] == 2).mean() * 100
        
        risk_increase = zero_failure - nonzero_failure
        
        print(f"     {col}:")
        print(f"        Zero (n={len(zero_group):,}):     {zero_failure:.2f}% failure, {zero_repair:.2f}% repair")
        print(f"        Non-Zero (n={len(nonzero_group):,}): {nonzero_failure:.2f}% failure, {nonzero_repair:.2f}% repair")
        print(f"        → {'✅ CONFIRMED' if risk_increase > 0 else '❌ NOT CONFIRMED'}: "
              f"Zero increases failure by {risk_increase:+.2f}%\n")
        
        zero_risk_tests.append({
            'Feature': col,
            'Zero_Failure_Rate': zero_failure,
            'NonZero_Failure_Rate': nonzero_failure,
            'Risk_Increase': risk_increase,
            'Confirmed': risk_increase > 0
        })
    
    results['zero_risk_tests'] = pd.DataFrame(zero_risk_tests)
    
    return results

def _plot_age_analysis(df: pd.DataFrame, target_col: str) -> None:
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    ax1 = axes[0]
    for status in [0, 1, 2]:
        subset = df[df[target_col] == status]['pump_age']
        ax1.hist(subset, bins=30, alpha=0.5, label=CLASS_LABELS[status], density=True)
    
    ax1.set_xlabel('Pump Age (years)')
    ax1.set_ylabel('Density')
    ax1.set_title('Pump Age Distribution by Status')
    ax1.legend()
    
    ax2 = axes[1]
    age_stats = df.groupby('pump_age').apply(
        lambda x: (x[target_col] == 1).mean() * 100
    ).reset_index()
    age_stats.columns = ['pump_age', 'failure_rate']
    
    ax2.scatter(age_stats['pump_age'], age_stats['failure_rate'], alpha=0.5, s=10)
    ax2.axhline(y=df[target_col].eq(1).mean() * 100, color='red', linestyle='--', 
                label=f'Global Avg: {df[target_col].eq(1).mean()*100:.1f}%')
    ax2.set_xlabel('Pump Age (years)')
    ax2.set_ylabel('Failure Rate (%)')
    ax2.set_title('Failure Rate vs Pump Age')
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig('pump_age_analysis.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"     ✓ Saved: pump_age_analysis.png")

def analyze_geospatial_signals(
    df: pd.DataFrame,
    target_col: str = 'status_group'
) -> Dict:
    print("\n" + "=" * 70)
    print("  ANALYSIS 4: Geospatial Intelligence")
    print("=" * 70)
    
    results = {}
    
    if 'region' in df.columns:
        region_stats = df.groupby('region').apply(
            lambda x: pd.Series({
                'count': len(x),
                'failure_rate': (x[target_col] == 1).mean(),
                'repair_rate': (x[target_col] == 2).mean()
            })
        ).reset_index()
        
        region_stats = region_stats.sort_values('failure_rate', ascending=False)
        
        hotspots = region_stats[region_stats['failure_rate'] > HOTSPOT_FAILURE_THRESHOLD]
        
        print(f"\n  📊 REGIONAL FAILURE ANALYSIS:")
        print("-" * 50)
        print(f"     Total Regions: {len(region_stats)}")
        print(f"     Hotspot Threshold: {HOTSPOT_FAILURE_THRESHOLD * 100:.0f}% failure rate")
        print(f"     Hotspot Regions Found: {len(hotspots)}")
        
        if len(hotspots) > 0:
            print(f"\n     🔴 HOTSPOT REGIONS (>{HOTSPOT_FAILURE_THRESHOLD * 100:.0f}% failure):")
            for _, row in hotspots.iterrows():
                print(f"        → {row['region']}: {row['failure_rate']*100:.1f}% failure "
                      f"(n={int(row['count']):,})")
        
        print(f"\n     📊 Top 10 Highest Failure Regions:")
        for _, row in region_stats.head(10).iterrows():
            status = "🔴" if row['failure_rate'] > HOTSPOT_FAILURE_THRESHOLD else "🟡" if row['failure_rate'] > 0.4 else "🟢"
            print(f"        {status} {row['region']}: {row['failure_rate']*100:.1f}% "
                  f"(n={int(row['count']):,})")
        
        results['region_stats'] = region_stats
        results['hotspots'] = hotspots
        
        _plot_regional_failure(region_stats)
    
    return results

def _plot_regional_failure(region_stats: pd.DataFrame) -> None:
    
    fig, ax = plt.subplots(figsize=(14, 8))
    
    data = region_stats.sort_values('failure_rate', ascending=True)
    
    colors = ['#e74c3c' if x > HOTSPOT_FAILURE_THRESHOLD else '#3498db' 
              for x in data['failure_rate']]
    
    bars = ax.barh(data['region'], data['failure_rate'] * 100, color=colors)
    
    ax.axvline(x=HOTSPOT_FAILURE_THRESHOLD * 100, color='red', linestyle='--', 
               linewidth=2, label=f'Hotspot Threshold ({HOTSPOT_FAILURE_THRESHOLD*100:.0f}%)')
    ax.axvline(x=data['failure_rate'].mean() * 100, color='orange', linestyle='--',
               linewidth=2, label=f'Mean ({data["failure_rate"].mean()*100:.1f}%)')
    
    ax.set_xlabel('Failure Rate (%)')
    ax.set_ylabel('Region')
    ax.set_title('Regional Failure Rates - Hotspot Identification')
    ax.legend(loc='lower right')
    
    plt.tight_layout()
    plt.savefig('regional_failure_analysis.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"     ✓ Saved: regional_failure_analysis.png")

# ---------------------------------------------------------------------------
# Numerical feature analysis (report Section IV.E, Figures 5-7).
# I run this on the RAW training values on purpose: the cleaning step turns the
# physically-invalid zeros (longitude=0, gps_height=0) into NaN and casts
# region_code/district_code to strings, which would hide the very things the
# report describes (the longitude spike at 0 and the region/district
# correlation). Reading raw keeps those numeric so the figures match the report.
# ---------------------------------------------------------------------------
RAW_TRAIN_VALUES_CANDIDATES = [
    'data/training_set_values.csv',
    'data/Training set values.csv',
    'training_set_values.csv',
    'Training set values.csv',
]


def _find_raw_train_values():
    for path in RAW_TRAIN_VALUES_CANDIDATES:
        if os.path.exists(path):
            return path
    return None


def _plot_numeric_distributions(df: pd.DataFrame, columns: List[str]) -> None:
    n_cols = len(columns)
    n_rows = (n_cols + 2) // 3

    fig1, axes1 = plt.subplots(n_rows, 3, figsize=(15, 4 * n_rows))
    axes1 = axes1.flatten()
    fig1.suptitle('Distribution Analysis: Histograms', fontsize=14, fontweight='bold', y=1.02)
    for i, col in enumerate(columns):
        ax = axes1[i]
        data = df[col].dropna()
        sns.histplot(data=data, kde=True, ax=ax, color='steelblue', edgecolor='white')
        ax.axvline(data.mean(), color='red', linestyle='--', label=f'Mean: {data.mean():.2f}')
        ax.axvline(data.median(), color='green', linestyle='-', label=f'Median: {data.median():.2f}')
        ax.set_title(f'{col}\nSkew: {data.skew():.2f}', fontsize=10)
        ax.legend(fontsize=8)
    for i in range(n_cols, len(axes1)):
        axes1[i].set_visible(False)
    plt.tight_layout()
    plt.savefig('distribution_histograms.png', dpi=150, bbox_inches='tight')
    plt.close()

    fig2, axes2 = plt.subplots(n_rows, 3, figsize=(15, 4 * n_rows))
    axes2 = axes2.flatten()
    fig2.suptitle('Outlier Analysis: Boxplots', fontsize=14, fontweight='bold', y=1.02)
    for i, col in enumerate(columns):
        ax = axes2[i]
        sns.boxplot(x=df[col].dropna(), ax=ax, color='lightcoral')
        ax.set_title(f'{col}', fontsize=10)
        ax.set_xlabel('')
    for i in range(n_cols, len(axes2)):
        axes2[i].set_visible(False)
    plt.tight_layout()
    plt.savefig('distribution_boxplots.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("   ✓ Saved: distribution_histograms.png, distribution_boxplots.png")


def _plot_numeric_correlation_heatmap(corr_matrix: pd.DataFrame) -> None:
    plt.figure(figsize=(10, 8))
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    sns.heatmap(
        corr_matrix, mask=mask,
        annot=True if len(corr_matrix) <= 12 else False,
        cmap='RdBu_r', center=0, vmin=-1, vmax=1, square=True,
        linewidths=0.5, fmt='.2f',
        cbar_kws={'shrink': 0.8, 'label': 'Correlation Coefficient'},
    )
    plt.title('Feature Correlation Heatmap', fontsize=14, fontweight='bold', pad=20)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig('correlation_heatmap.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("   ✓ Saved: correlation_heatmap.png")


def analyze_numerical_features() -> pd.DataFrame:
    print("\n" + "=" * 70)
    print("  ANALYSIS 5: Numerical Feature Distributions (Figures 5-7)")
    print("=" * 70)

    raw_path = _find_raw_train_values()
    if raw_path is None:
        print("   ⚠️  Raw training values not found; skipping numerical EDA.")
        print(f"      Looked for: {RAW_TRAIN_VALUES_CANDIDATES}")
        return None

    print(f"   Loading raw values: {raw_path}")
    raw = pd.read_csv(raw_path)

    numeric_cols = raw.select_dtypes(include=[np.number]).columns.tolist()
    if 'id' in numeric_cols:
        numeric_cols.remove('id')

    print(f"   Numeric features analysed ({len(numeric_cols)}): {numeric_cols}")

    _plot_numeric_distributions(raw, numeric_cols[:9])

    corr_matrix = raw[numeric_cols].corr()
    _plot_numeric_correlation_heatmap(corr_matrix)

    if 'region_code' in numeric_cols and 'district_code' in numeric_cols:
        rho = corr_matrix.loc['region_code', 'district_code']
        print(f"   region_code <-> district_code correlation: rho = {rho:.2f}")

    return corr_matrix


def run_strategic_eda():
    print("\n" + "=" * 70)
    print("  STRATEGIC RISK-ORIENTED EDA")
    print("=" * 70)

    train_values = pd.read_csv("cleaned_train_values.csv")
    train_labels = pd.read_csv("cleaned_train_labels.csv")

    if "status_group" not in train_labels.columns:
        raise ValueError("cleaned_train_labels.csv must contain a status_group column.")

    df = train_values.copy()
    df["status_group"] = train_labels["status_group"].values

    results = {
        "sure_kill": analyze_sure_kill_signals(df, SURE_KILL_FEATURES),
        "hidden_risk": analyze_hidden_risk_signals(df, HIDDEN_RISK_FEATURES),
        "temporal_physical": analyze_temporal_physical_signals(df),
        "geospatial": analyze_geospatial_signals(df),
    }

    try:
        results["numerical"] = analyze_numerical_features()
    except Exception as e:
        print(f"   ⚠️  Numerical feature analysis skipped: {e}")

    print("\nStrategic EDA complete.")
    print("Generated: hidden_risk_comparison.png, pump_age_analysis.png, regional_failure_analysis.png,")
    print("           distribution_histograms.png, distribution_boxplots.png, correlation_heatmap.png")
    return results


if __name__ == "__main__":
    run_strategic_eda()
