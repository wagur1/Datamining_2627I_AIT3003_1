"""
Data Mining (AIT3003) - Problem Set 1: Data Understanding
Solutions and reproducible calculations for all exercises.
"""

import os
import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt

def solve_problem_1():
    print("=" * 60)
    print("PROBLEM 1: Age Attribute Analysis")
    print("=" * 60)
    age_data = np.array([13, 15, 16, 16, 19, 20, 20, 21, 22, 22, 25, 25, 25, 25, 30,
                         33, 33, 35, 35, 35, 35, 36, 40, 45, 46, 52, 70])
    n = len(age_data)
    mean_val = np.mean(age_data)
    median_val = np.median(age_data)
    
    vals, counts = np.unique(age_data, return_counts=True)
    max_count = np.max(counts)
    modes = vals[counts == max_count]
    
    midrange_val = (np.min(age_data) + np.max(age_data)) / 2.0
    
    # Quartiles (Tukey's hinges)
    q1 = age_data[6]   # 7th value
    q3 = age_data[20]  # 21st value
    iqr = q3 - q1
    
    var_sample = np.var(age_data, ddof=1)
    std_sample = np.std(age_data, ddof=1)
    var_pop = np.var(age_data, ddof=0)
    std_pop = np.std(age_data, ddof=0)
    
    print(f"a. Mean: {mean_val:.4f} (Sum = {np.sum(age_data)} / {n}), Median: {median_val}")
    print(f"b. Mode: {modes} (count = {max_count}), Modality: Bimodal")
    print(f"c. Midrange: {midrange_val}")
    print(f"d. Q1: {q1}, Q3: {q3}, IQR: {iqr}")
    print(f"e. Five-number summary: [Min={np.min(age_data)}, Q1={q1}, Median={median_val}, Q3={q3}, Max={np.max(age_data)}]")
    print(f"f. Outlier boundaries: [{q1 - 1.5*iqr}, {q3 + 1.5*iqr}] -> Outlier: 70")
    print(f"g. Sample Variance: {var_sample:.4f}, Sample Std: {std_sample:.4f}")
    print(f"   Population Variance: {var_pop:.4f}, Population Std: {std_pop:.4f}")

def solve_problem_2():
    print("\n" + "=" * 60)
    print("PROBLEM 2: Age and %Fat Correlation Analysis")
    print("=" * 60)
    age = np.array([23, 23, 27, 27, 39, 41, 47, 49, 50, 52, 54, 54, 56, 57, 58, 58, 60, 61], dtype=float)
    fat = np.array([9.5, 26.5, 7.8, 17.8, 31.4, 25.9, 27.4, 27.2, 31.2, 34.6, 42.5, 28.8, 33.4, 30.2, 34.1, 32.9, 41.2, 35.7], dtype=float)
    
    print("a. Age statistics:")
    print(f"   Mean: {np.mean(age):.4f}, Median: {np.median(age):.4f}")
    print(f"   Sample Var: {np.var(age, ddof=1):.4f}, Sample Std: {np.std(age, ddof=1):.4f}")
    print(f"   Pop Var: {np.var(age, ddof=0):.4f}, Pop Std: {np.std(age, ddof=0):.4f}")
    
    print("   %Fat statistics:")
    print(f"   Mean: {np.mean(fat):.4f}, Median: {np.median(fat):.4f}")
    print(f"   Sample Var: {np.var(fat, ddof=1):.4f}, Sample Std: {np.std(fat, ddof=1):.4f}")
    print(f"   Pop Var: {np.var(fat, ddof=0):.4f}, Pop Std: {np.std(fat, ddof=0):.4f}")
    
    r_pearson, p_p = stats.pearsonr(age, fat)
    r_spearman, p_s = stats.spearmanr(age, fat)
    print(f"d. Pearson Correlation Coefficient: {r_pearson:.4f}")
    print(f"   Spearman Rank Correlation: {r_spearman:.4f}")

def solve_problem_3():
    print("\n" + "=" * 60)
    print("PROBLEM 3: Cristiano Ronaldo UCL Analysis (03/04 to 21/22)")
    print("=" * 60)
    data = [
        [5, 0, 1], [7, 0, 2], [6, 0, 0], [11, 3, 5], [11, 8, 1],
        [12, 4, 3], [6, 7, 1], [12, 6, 4], [10, 10, 3], [12, 12, 1],
        [11, 17, 5], [12, 10, 4], [12, 16, 4], [13, 12, 6], [13, 15, 3],
        [9, 6, 2], [8, 4, 1], [6, 4, 2], [7, 6, 0]
    ]
    df = pd.DataFrame(data, columns=['Appearances', 'Goals', 'Assists'])
    for col in df.columns:
        vals = df[col].values
        v, c = np.unique(vals, return_counts=True)
        m = v[c == np.max(c)]
        print(f"{col}:")
        print(f"   Mean: {np.mean(vals):.4f}, Median: {np.median(vals)}, Mode: {m}")
        print(f"   Sample Var: {np.var(vals, ddof=1):.4f}, Sample Std: {np.std(vals, ddof=1):.4f}")
        print(f"   Pop Var: {np.var(vals, ddof=0):.4f}, Pop Std: {np.std(vals, ddof=0):.4f}")
        sorted_v = np.sort(vals)
        print(f"   Q1: {sorted_v[4]}, Median: {sorted_v[9]}, Q3: {sorted_v[14]}")
    
    print("\nSample Covariance Matrix (ddof=1):")
    print(df.cov().round(4))
    print("\nPopulation Covariance Matrix (ddof=0):")
    print(df.cov(ddof=0).round(4))

def solve_problem_4():
    print("\n" + "=" * 60)
    print("PROBLEM 4: Pearson's Chi-Square Test (Games vs Study Major)")
    print("=" * 60)
    obs = np.array([
        [130, 100],  # Math
        [35, 165],   # History
        [280, 90]    # CS
    ])
    row_totals = obs.sum(axis=1)
    col_totals = obs.sum(axis=0)
    grand_total = obs.sum()
    
    expected = np.outer(row_totals, col_totals) / grand_total
    print("a. Expected Frequencies:")
    exp_df = pd.DataFrame(expected, index=['Math', 'History', 'CS'], columns=['Like games', 'Do not like games'])
    print(exp_df.round(4))
    
    chi2_val, p_val, dof, _ = stats.chi2_contingency(obs, correction=False)
    print(f"\nb. Chi-Square Statistic: {chi2_val:.4f}")
    print(f"   Degrees of Freedom: {dof}")
    print(f"   p-value: {p_val:.4e}")
    crit_05 = stats.chi2.ppf(0.95, df=dof)
    crit_01 = stats.chi2.ppf(0.99, df=dof)
    print(f"   Critical value (alpha=0.05): {crit_05:.4f}")
    print(f"   Critical value (alpha=0.01): {crit_01:.4f}")
    print(f"   Conclusion: Reject H0 -> Significant correlation exists.")

if __name__ == "__main__":
    solve_problem_1()
    solve_problem_2()
    solve_problem_3()
    solve_problem_4()
