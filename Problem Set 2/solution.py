"""
Data Mining (AIT3003) - Problem Set 2: Data Preparation
Solutions, reproducible calculations, and visualizations for all exercises.
"""

import os
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
import numpy as np


def solve_problem_1():
    print("=" * 70)
    print("PROBLEM 1: Handling Missing Values in Real-World Data")
    print("=" * 70)
    print(
        """
Summary of core methods described in Data Mining (Han, Kamber & Pei):
1. Ignore the tuple (Listwise / Case Deletion):
   - Useful when class label is missing in supervised classification tasks.
   - Poor performance when missingness varies widely or sample size is small.
2. Fill in missing values manually:
   - High accuracy via domain experts, but infeasible for large-scale datasets.
3. Use a global constant (e.g., 'Unknown', -999, 'N/A'):
   - Simple and transparent, but data mining algorithms may treat the constant
     as an informative real value/cluster.
4. Use a measure of central tendency (Mean or Median):
   - Mean for symmetric/normal distributions.
   - Median for skewed distributions with outliers.
5. Use class-conditional attribute mean or median:
   - Impute missing values using the mean/median of tuples sharing the same class label.
6. Use the most probable value (Predictive / Model-based Imputation):
   - Regression models, decision trees, or k-Nearest Neighbors (k-NN) to predict
     the missing attribute based on other correlated attributes.
7. Advanced / Multiple Imputation (e.g., MICE, missForest):
   - Generates multiple completed datasets to capture imputation uncertainty.
"""
    )


def solve_problem_2(output_dir: str = "images"):
    print("\n" + "=" * 70)
    print("PROBLEM 2: Data Smoothing & Outlier Detection for Age Attribute")
    print("=" * 70)

    # Given data sorted
    age_data = np.array(
        [
            13, 15, 16, 16, 19, 20, 20, 21, 22,
            22, 25, 25, 25, 25, 30, 33, 33, 35,
            35, 35, 35, 36, 40, 45, 46, 52, 70,
        ],
        dtype=float,
    )
    n = len(age_data)
    bin_depth = 3
    num_bins = n // bin_depth

    print(f"Total records: n = {n}")
    print(f"Equal-frequency bin depth: {bin_depth} -> Number of bins: {num_bins}")
    print("-" * 50)

    # (a) Smoothing by bin means
    bins = age_data.reshape(num_bins, bin_depth)
    bin_means = bins.mean(axis=1)
    bin_medians = np.median(bins, axis=1)

    # Bin boundaries smoothing
    bin_boundaries = np.zeros_like(bins)
    for i in range(num_bins):
        b_min = bins[i, 0]
        b_max = bins[i, -1]
        for j in range(bin_depth):
            val = bins[i, j]
            if abs(val - b_min) <= abs(val - b_max):
                bin_boundaries[i, j] = b_min
            else:
                bin_boundaries[i, j] = b_max

    smoothed_means = np.repeat(bin_means, bin_depth)
    smoothed_medians = np.repeat(bin_medians, bin_depth)
    smoothed_boundaries = bin_boundaries.flatten()

    print("(a) Smoothing by Bin Means (depth = 3):")
    print(
        f"{'Bin #':<6} | {'Original Values':<18} | {'Sum':<6} | {'Bin Mean (exact)':<18} | {'Bin Mean (round)':<16}"
    )
    print("-" * 72)
    for i in range(num_bins):
        b_vals = [int(x) for x in bins[i]]
        b_sum = sum(b_vals)
        b_mean = bin_means[i]
        print(
            f"Bin {i+1:<2} | {str(b_vals):<18} | {b_sum:<6} | {b_mean:<18.4f} | {b_mean:<16.2f}"
        )

    print("\nSmoothed series by bin means:")
    print([round(float(x), 2) for x in smoothed_means])

    print(
        "\nComment on effect: Local fluctuations and noise are mitigated while the monotonic trend is preserved."
    )
    print(
        "Notice that the extreme outlier 70 in Bin 9 is pulled down to 56.00 by 46 and 52."
    )
    print(
        "Bin 7 [35, 35, 35] has zero internal variance, so its mean remains exactly 35.00."
    )

    # (b) Outlier detection
    print("\n" + "-" * 50)
    print("(b) Determining Outliers:")
    q1 = age_data[6]  # 7th value
    q3 = age_data[20]  # 21st value
    iqr = q3 - q1
    lower_fence = q1 - 1.5 * iqr
    upper_fence = q3 + 1.5 * iqr
    extreme_upper = q3 + 3.0 * iqr

    mean_val = np.mean(age_data)
    std_sample = np.std(age_data, ddof=1)
    z_scores = (age_data - mean_val) / std_sample

    mad_mean = np.mean(np.abs(age_data - mean_val))
    median_val = np.median(age_data)
    mad_median = np.median(np.abs(age_data - median_val))
    mod_z_scores = 0.6745 * np.abs(age_data - median_val) / mad_median

    print(f"1. Tukey's IQR Method (Boxplot rule):")
    print(f"   Q1 = {q1}, Q3 = {q3}, IQR = {iqr}")
    print(f"   Inner Fences: [{lower_fence:.1f}, {upper_fence:.1f}]")
    print(f"   Outer Fences: [{q1 - 3.0*iqr:.1f}, {extreme_upper:.1f}]")
    outliers_iqr = age_data[
        (age_data < lower_fence) | (age_data > upper_fence)
    ]
    print(f"   Detected Outliers: {outliers_iqr.astype(int).tolist()} (70 > 57.5)")

    print(f"2. Z-Score (3-Sigma) Rule:")
    print(f"   Sample Mean = {mean_val:.4f}, Sample Std = {std_sample:.4f}")
    print(f"   Z-score for 52 = {z_scores[-2]:.4f}")
    print(f"   Z-score for 70 = {z_scores[-1]:.4f} (> 3.0 -> Outlier!)")

    print(f"3. Modified Z-Score using MAD:")
    print(f"   Median = {median_val:.1f}, MAD = {mad_median:.1f}")
    print(
        f"   Modified Z for 70 = {mod_z_scores[-1]:.4f} (> 3.5 -> Extreme Outlier!)"
    )

    # (c) Other smoothing methods
    print("\n" + "-" * 50)
    print("(c) Other Data Smoothing Methods:")
    print(
        "1. Smoothing by bin medians: Replaces values by the median of their bin."
    )
    print(
        "2. Smoothing by bin boundaries: Replaces values with the nearest boundary (min/max)."
    )
    print("3. Regression: Fitting linear or polynomial functions (y = wx + b).")
    print("4. Clustering: Organizing data into clusters and using centroids.")
    print("5. Moving average / Kernel smoothing: Rolling window averaging.")
    print(
        "6. Wavelet transform / Spectral filtering: Filtering high-frequency noise."
    )

    # Generate visualization for Problem 2
    os.makedirs(output_dir, exist_ok=True)

    # Plot 1: Bin smoothing comparison
    fig, axes = plt.subplots(2, 1, figsize=(11, 9))

    x = np.arange(1, n + 1)
    axes[0].plot(
        x,
        age_data,
        "o-",
        color="#1f77b4",
        linewidth=2,
        label="Original Age Data",
        markersize=6,
    )
    axes[0].step(
        x,
        smoothed_means,
        where="mid",
        color="#d62728",
        linewidth=2.5,
        linestyle="--",
        label="Smoothed by Bin Means (depth=3)",
    )
    axes[0].set_title(
        "Data Smoothing by Bin Means (Bin Depth = 3)",
        fontsize=14,
        fontweight="bold",
    )
    axes[0].set_xlabel("Sorted Data Index", fontsize=12)
    axes[0].set_ylabel("Age", fontsize=12)
    axes[0].grid(True, linestyle=":", alpha=0.6)
    axes[0].legend(fontsize=11)

    # Add bin dividers
    for b_idx in range(1, num_bins):
        axes[0].axvline(
            b_idx * bin_depth + 0.5, color="gray", linestyle=":", alpha=0.5
        )

    # Subplot 2: Comparison of smoothing techniques
    axes[1].plot(
        x,
        age_data,
        "o-",
        color="#1f77b4",
        linewidth=1.5,
        alpha=0.6,
        label="Original",
    )
    axes[1].plot(
        x,
        smoothed_means,
        "s-",
        color="#d62728",
        linewidth=1.8,
        label="Bin Means",
    )
    axes[1].plot(
        x,
        smoothed_medians,
        "^-",
        color="#2ca02c",
        linewidth=1.8,
        label="Bin Medians",
    )
    axes[1].plot(
        x,
        smoothed_boundaries,
        "x-",
        color="#ff7f0e",
        linewidth=1.8,
        label="Bin Boundaries",
    )
    axes[1].set_title(
        "Comparison: Bin Means vs. Bin Medians vs. Bin Boundaries",
        fontsize=14,
        fontweight="bold",
    )
    axes[1].set_xlabel("Sorted Data Index", fontsize=12)
    axes[1].set_ylabel("Age", fontsize=12)
    axes[1].grid(True, linestyle=":", alpha=0.6)
    axes[1].legend(fontsize=11)

    plt.tight_layout()
    fig.savefig(
        os.path.join(output_dir, "p2_bin_smoothing.png"),
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)

    # Plot 2: Outlier detection
    fig, (ax_box, ax_z) = plt.subplots(1, 2, figsize=(12, 5))

    # Boxplot
    box = ax_box.boxplot(
        age_data,
        patch_artist=True,
        boxprops=dict(facecolor="#c6dbef", color="#08519c"),
        whiskerprops=dict(color="#08519c", linewidth=1.5),
        capprops=dict(color="#08519c", linewidth=1.5),
        medianprops=dict(color="#d94801", linewidth=2),
        flierprops=dict(
            marker="o", markerfacecolor="red", markeredgecolor="darkred", markersize=8
        ),
    )
    ax_box.axhline(
        upper_fence,
        color="red",
        linestyle="--",
        linewidth=1.2,
        label=f"Upper Fence = {upper_fence:.1f}",
    )
    ax_box.axhline(
        lower_fence,
        color="green",
        linestyle="--",
        linewidth=1.2,
        label=f"Lower Fence = {lower_fence:.1f}",
    )
    ax_box.text(
        1.05,
        70,
        " Outlier: 70",
        color="red",
        fontweight="bold",
        verticalalignment="center",
    )
    ax_box.set_title(
        "Tukey's Boxplot Outlier Detection", fontsize=13, fontweight="bold"
    )
    ax_box.set_ylabel("Age", fontsize=12)
    ax_box.set_xticks([1])
    ax_box.set_xticklabels(["Age Attribute"])
    ax_box.legend(loc="upper left")
    ax_box.grid(True, linestyle=":", alpha=0.5)

    # Z-scores
    ax_z.stem(
        x,
        z_scores,
        linefmt="b-",
        markerfmt="bo",
        basefmt="k-",
    )
    ax_z.axhline(
        3.0,
        color="red",
        linestyle="--",
        linewidth=1.5,
        label="3-Sigma Threshold (z = 3.0)",
    )
    ax_z.axhline(
        -3.0,
        color="red",
        linestyle="--",
        linewidth=1.5,
    )
    ax_z.axhline(
        2.0,
        color="orange",
        linestyle=":",
        linewidth=1.2,
        label="2-Sigma Threshold (z = 2.0)",
    )
    ax_z.text(
        27,
        z_scores[-1],
        f" z = {z_scores[-1]:.2f}",
        color="red",
        fontweight="bold",
        verticalalignment="bottom",
        horizontalalignment="right",
    )
    ax_z.set_title(
        "Z-Score Outlier Analysis (Mean=29.96, Std=12.94)",
        fontsize=13,
        fontweight="bold",
    )
    ax_z.set_xlabel("Sorted Data Index", fontsize=12)
    ax_z.set_ylabel("Z-Score", fontsize=12)
    ax_z.legend(loc="upper left")
    ax_z.grid(True, linestyle=":", alpha=0.5)

    plt.tight_layout()
    fig.savefig(
        os.path.join(output_dir, "p2_outlier_detection.png"),
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)

    print(
        f"Visualizations saved to {output_dir}/p2_bin_smoothing.png and {output_dir}/p2_outlier_detection.png"
    )


def solve_problem_3():
    print("\n" + "=" * 70)
    print("PROBLEM 3: Issues to Consider During Data Integration")
    print("=" * 70)
    print(
        """
Core issues in Data Integration (Han, Kamber & Pei, Section 3.3):

1. Schema Integration and Entity Identification:
   - Entity Identification Problem: Matching real-world entities across multiple databases
     (e.g., cust_id in DB1 == customer_number in DB2).
   - Attribute Naming Conflicts:
     * Synonyms: Different names for the same attribute (e.g., fee vs charge).
     * Homonyms: Same name for different attributes (e.g., address for home vs billing).
   - Structural Inconsistencies: Same information modeled as an attribute in one schema
     and as a separate table/relation in another.
   - Solution: Metadata analysis, data dictionaries, schema matching tools.

2. Redundancy and Correlation Analysis:
   - An attribute is redundant if it can be derived from others (e.g., age from birth_date,
     or total_price from unit_price * quantity).
   - Detection methods:
     * Nominal attributes: Chi-Square Test of Independence (chi^2).
     * Numeric attributes: Pearson Correlation Coefficient (r) or Covariance (Cov(A,B)).
   - Resolving redundancy reduces storage waste and avoids algorithmic bias (multicollinearity).

3. Tuple Duplication:
   - Repeated records referring to the same entity (e.g., slight typo in name/address).
   - Solution: Record linkage, fuzzy deduplication, string distance metrics (Levenshtein, Jaro-Winkler).

4. Data Value Conflicts and Detection:
   - For the same entity, attribute values from different sources may conflict.
   - Causes:
     * Representation differences: Metric vs Imperial (kg vs lb, Celsius vs Fahrenheit).
     * Currency differences: USD vs EUR vs VND.
     * Different semantic encodings: 1/0 vs True/False vs M/F.
     * Temporal divergence: Different snapshot timestamps / update frequencies.
   - Solution: Standardization, transformation pipelines, master data governance rules.
"""
    )


def solve_problem_4():
    print("\n" + "=" * 70)
    print("PROBLEM 4: Value Ranges of Normalization Methods")
    print("=" * 70)
    print(
        """
(a) Min-Max Normalization:
    Formula: v' = ((v - min_A) / (max_A - min_A)) * (new_max_A - new_min_A) + new_min_A
    Value Range: [new_min_A, new_max_A]
    Specifically, when mapped onto [0.0, 1.0], the range is exactly [0.0, 1.0].
    (Note: If applied to future out-of-sample data points that exceed the original
    bounds, an 'out-of-bounds error' occurs unless values are clipped).

(b) Z-Score Normalization:
    Formula: v' = (v - mean_A) / sigma_A
    Value Range: (-infinity, +infinity)
    (The entire real number line R. In typical empirical data, ~99.73% of values
    fall within [-3, +3], but there is no mathematical upper or lower bound).

(c) Z-Score Normalization using Mean Absolute Deviation (MAD):
    Formula: v' = (v - mean_A) / s_A, where s_A = (1/n) * sum(|v_i - mean_A|)
    Value Range: (-infinity, +infinity)
    (The entire real number line R. Similar to standard z-score, but more robust
    against outliers because deviations are not squared).

(d) Normalization by Decimal Scaling:
    Formula: v' = v / (10^j), where j is the smallest integer such that max(|v'|) < 1.
    Value Range: (-1.0, 1.0)
    (Strictly open interval (-1, 1). If all values in the attribute are non-negative,
    the range is [0.0, 1.0)).
"""
    )


def solve_problem_5(output_dir: str = "images"):
    print("\n" + "=" * 70)
    print("PROBLEM 5: Normalization Calculations on Age Dataset for v = 35")
    print("=" * 70)

    age_data = np.array(
        [
            13, 15, 16, 16, 19, 20, 20, 21, 22,
            22, 25, 25, 25, 25, 30, 33, 33, 35,
            35, 35, 35, 36, 40, 45, 46, 52, 70,
        ],
        dtype=float,
    )
    v = 35.0
    min_age = float(np.min(age_data))
    max_age = float(np.max(age_data))
    sum_age = float(np.sum(age_data))
    n = len(age_data)
    mean_exact = sum_age / n  # 809 / 27 = 29.96296...
    mean_rounded = 29.96
    std_given = 12.94

    # (a) Min-Max Normalization onto [0.0, 1.0]
    min_max_35 = (v - min_age) / (max_age - min_age)
    print("(a) Min-Max Normalization onto [0.0, 1.0]:")
    print(f"    min = {min_age}, max = {max_age}")
    print(f"    v' = (35 - 13) / (70 - 13) = 22 / 57")
    print(f"    v' = {min_max_35:.6f} ≈ {min_max_35:.4f}")

    # (b) Z-Score Normalization
    z_score_given = (v - mean_rounded) / std_given
    z_score_exact = (v - mean_exact) / std_given
    print("\n(b) Z-Score Normalization (std = 12.94 years):")
    print(
        f"    Using rounded mean (29.96): v' = (35 - 29.96) / 12.94 = 5.04 / 12.94 = {z_score_given:.6f} ≈ {z_score_given:.4f}"
    )
    print(
        f"    Using exact mean (29.9630): v' = (35 - 29.9630) / 12.94 = {z_score_exact:.6f} ≈ {z_score_exact:.4f}"
    )

    # (c) Decimal Scaling
    # Smallest j such that max(|v|) / 10^j < 1
    # max = 70 -> 70 / 10^2 = 0.70 < 1 -> j = 2
    j = int(np.ceil(np.log10(max_age + 1)))
    decimal_scaled_35 = v / (10**j)
    print("\n(c) Normalization by Decimal Scaling:")
    print(f"    max(|age|) = {max_age}")
    print(f"    Smallest integer j such that 70 / 10^j < 1 is j = 2 (10^2 = 100)")
    print(
        f"    v' = 35 / 10^2 = 35 / 100 = {decimal_scaled_35:.4f}"
    )

    # (d) Method preference commentary
    print("\n(d) Method Preference and Reasons:")
    print(
        """
    Preference: Z-Score Normalization (or Z-score using MAD).
    Reasons:
    1. Outlier Sensitivity:
       - The dataset contains an extreme outlier: age 70 (upper fence is 57.5).
       - In Min-Max normalization, the outlier 70 stretches the denominator (70 - 13 = 57).
         Consequently, 26 out of 27 data points (values 13 to 52) are heavily compressed
         into the narrow sub-range [0.0, 0.684], leaving the upper third of the interval empty.
       - Decimal scaling is similarly dominated by the maximum value (j=2) and does not
         center the data or account for data variance.
    2. Prevention of Out-of-Bounds Errors:
       - Min-Max encounters out-of-bounds errors if future data points arrive outside [13, 70].
       - Z-score normalization has no fixed boundaries, allowing it to naturally accommodate
         unseen values.
    3. Distribution Alignment:
       - Z-score centers the distribution at 0 and measures variation in standard deviations,
         making it ideal for subsequent statistical and machine learning algorithms
         (e.g., gradient descent, PCA, distance-based classifiers).
"""
    )

    # Generate visualization for Problem 5
    os.makedirs(output_dir, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # All normalized arrays
    norm_minmax = (age_data - min_age) / (max_age - min_age)
    norm_zscore = (age_data - mean_exact) / std_given
    norm_decimal = age_data / (10**j)

    # Subplot 1: Original Distribution
    axes[0, 0].hist(
        age_data,
        bins=10,
        color="#1f77b4",
        edgecolor="black",
        alpha=0.7,
        rwidth=0.85,
    )
    axes[0, 0].axvline(
        35,
        color="red",
        linestyle="--",
        linewidth=2,
        label="Target Value (35)",
    )
    axes[0, 0].axvline(
        mean_exact,
        color="green",
        linestyle=":",
        linewidth=2,
        label=f"Mean ({mean_exact:.1f})",
    )
    axes[0, 0].set_title(
        "Original Age Distribution", fontsize=13, fontweight="bold"
    )
    axes[0, 0].set_xlabel("Age", fontsize=11)
    axes[0, 0].set_ylabel("Frequency", fontsize=11)
    axes[0, 0].legend()
    axes[0, 0].grid(True, linestyle=":", alpha=0.5)

    # Subplot 2: Min-Max
    axes[0, 1].hist(
        norm_minmax,
        bins=10,
        color="#ff7f0e",
        edgecolor="black",
        alpha=0.7,
        rwidth=0.85,
    )
    axes[0, 1].axvline(
        min_max_35,
        color="red",
        linestyle="--",
        linewidth=2,
        label=f"v' = {min_max_35:.4f}",
    )
    axes[0, 1].set_title(
        "Min-Max Normalized [0.0, 1.0]", fontsize=13, fontweight="bold"
    )
    axes[0, 1].set_xlabel("Normalized Value", fontsize=11)
    axes[0, 1].legend()
    axes[0, 1].grid(True, linestyle=":", alpha=0.5)

    # Subplot 3: Z-Score
    axes[1, 0].hist(
        norm_zscore,
        bins=10,
        color="#2ca02c",
        edgecolor="black",
        alpha=0.7,
        rwidth=0.85,
    )
    axes[1, 0].axvline(
        z_score_exact,
        color="red",
        linestyle="--",
        linewidth=2,
        label=f"v' = {z_score_exact:.4f}",
    )
    axes[1, 0].set_title("Z-Score Normalized", fontsize=13, fontweight="bold")
    axes[1, 0].set_xlabel("Normalized Value (Z-Score)", fontsize=11)
    axes[1, 0].set_ylabel("Frequency", fontsize=11)
    axes[1, 0].legend()
    axes[1, 0].grid(True, linestyle=":", alpha=0.5)

    # Subplot 4: Decimal Scaling
    axes[1, 1].hist(
        norm_decimal,
        bins=10,
        color="#9467bd",
        edgecolor="black",
        alpha=0.7,
        rwidth=0.85,
    )
    axes[1, 1].axvline(
        decimal_scaled_35,
        color="red",
        linestyle="--",
        linewidth=2,
        label=f"v' = {decimal_scaled_35:.4f}",
    )
    axes[1, 1].set_title(
        "Decimal Scaling Normalized (j=2)", fontsize=13, fontweight="bold"
    )
    axes[1, 1].set_xlabel("Normalized Value", fontsize=11)
    axes[1, 1].legend()
    axes[1, 1].grid(True, linestyle=":", alpha=0.5)

    plt.tight_layout()
    fig.savefig(
        os.path.join(output_dir, "p5_normalization_comparison.png"),
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)

    print(
        f"Visualization saved to {output_dir}/p5_normalization_comparison.png"
    )


# =====================================================================
# PROBLEM 6: Automatic Concept Hierarchy Generation Algorithm
# =====================================================================


class ConceptHierarchyGenerator:
    """
    Automatic concept hierarchy generator for nominal schema attributes
    based on the number of distinct values per attribute (Han & Kamber, Section 3.5.6).

    Principle:
    Concepts at higher abstraction levels cover broader categories and thus have
    fewer distinct values (e.g. country: 15). Concepts at lower levels are more
    specialized and have many distinct values (e.g. street: 674,339).
    """

    def __init__(self, schema_name: str):
        self.schema_name = schema_name
        self.attributes: Dict[str, int] = {}
        self.ordered_hierarchy: List[Tuple[str, int]] = []

    def add_attribute(self, attribute_name: str, distinct_count: int):
        """Add attribute with its distinct value count."""
        if distinct_count < 1:
            raise ValueError(
                f"Distinct count for {attribute_name} must be >= 1"
            )
        self.attributes[attribute_name] = distinct_count

    def generate_hierarchy(self) -> List[Tuple[str, int]]:
        """
        Sort attributes in ascending order of distinct counts.
        Ascending order = Top-Down hierarchy:
        Level 1 (Top / Root): Smallest distinct count
        Level M (Bottom / Leaf): Largest distinct count
        """
        if not self.attributes:
            raise ValueError("No attributes defined in schema.")

        # Sort by distinct count ascending
        self.ordered_hierarchy = sorted(
            self.attributes.items(), key=lambda item: item[1]
        )
        return self.ordered_hierarchy

    def display_hierarchy(self):
        """Print textual representation of the concept hierarchy."""
        if not self.ordered_hierarchy:
            self.generate_hierarchy()

        print(f"\nConcept Hierarchy for Dimension: '{self.schema_name}'")
        print("=" * 60)
        m = len(self.ordered_hierarchy)
        for level, (attr, count) in enumerate(self.ordered_hierarchy):
            pos_label = (
                "(Top Level - Most Generalized)"
                if level == 0
                else (
                    "(Bottom Level - Most Detailed)"
                    if level == m - 1
                    else "(Intermediate Level)"
                )
            )
            indent = "    " * level
            arrow = "└── " if level > 0 else "┌── "
            print(
                f"{indent}{arrow}Level {level + 1}: {attr:<20} [{count:>10,d} distinct values]  {pos_label}"
            )
        print("=" * 60)

        # Print partial order chain
        chain = " < ".join([attr for attr, _ in reversed(self.ordered_hierarchy)])
        print(f"Hierarchical Ordering (Specific < General):")
        print(f"  {chain}\n")

    def plot_hierarchy(self, output_path: str):
        """Plot graphical tree of the concept hierarchy."""
        if not self.ordered_hierarchy:
            self.generate_hierarchy()

        fig, ax = plt.subplots(figsize=(10, 6))

        m = len(self.ordered_hierarchy)
        y_positions = np.linspace(0.85, 0.15, m)
        x_position = 0.5

        colors = ["#2b5c8f", "#3c7bb6", "#6baed6", "#bdd7e7"]
        if m > len(colors):
            colors = plt.cm.Blues(np.linspace(0.9, 0.3, m))

        for i, (attr, count) in enumerate(self.ordered_hierarchy):
            y = y_positions[i]
            # Draw node box
            node_text = (
                f"Level {i+1}: {attr.upper()}\n({count:,d} distinct values)"
            )
            ax.text(
                x_position,
                y,
                node_text,
                ha="center",
                va="center",
                fontsize=11,
                fontweight="bold",
                color="white",
                bbox=dict(
                    boxstyle="round,pad=0.6",
                    facecolor=colors[i % len(colors)],
                    edgecolor="black",
                    linewidth=1.5,
                ),
            )

            # Draw connector arrow from previous level
            if i > 0:
                prev_y = y_positions[i - 1]
                ax.annotate(
                    "",
                    xy=(x_position, y + 0.05),
                    xytext=(x_position, prev_y - 0.05),
                    arrowprops=dict(
                        arrowstyle="->,head_width=0.5,head_length=0.7",
                        lw=2.5,
                        color="#333333",
                    ),
                )
                # Label relation
                ax.text(
                    x_position + 0.04,
                    (prev_y + y) / 2,
                    "specializes into",
                    fontsize=9,
                    color="#555555",
                    style="italic",
                    va="center",
                )

        # Annotations on sides
        ax.text(
            0.15,
            0.85,
            "▲ High-level concept\n  (Few distinct values,\n   broad abstraction)",
            ha="center",
            va="center",
            fontsize=10,
            color="#08519c",
            fontweight="bold",
            bbox=dict(
                boxstyle="square,pad=0.4",
                facecolor="#eff3ff",
                edgecolor="#08519c",
            ),
        )
        ax.text(
            0.15,
            0.15,
            "▼ Low-level concept\n  (Many distinct values,\n   fine granularity)",
            ha="center",
            va="center",
            fontsize=10,
            color="#08519c",
            fontweight="bold",
            bbox=dict(
                boxstyle="square,pad=0.4",
                facecolor="#eff3ff",
                edgecolor="#08519c",
            ),
        )

        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")
        ax.set_title(
            f"Concept Hierarchy for Nominal Schema: '{self.schema_name}'\nGenerated by Distinct Value Count Heuristic",
            fontsize=13,
            fontweight="bold",
            pad=15,
        )

        plt.tight_layout()
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close(fig)


def solve_problem_6(output_dir: str = "images"):
    print("\n" + "=" * 70)
    print("PROBLEM 6: Automatic Concept Hierarchy Generation for Nominal Data")
    print("=" * 70)

    # Example 1: Location dimension from textbook (Example 3.7, Han & Kamber)
    location_hierarchy = ConceptHierarchyGenerator("Location Dimension")
    location_hierarchy.add_attribute("country", 15)
    location_hierarchy.add_attribute("province_or_state", 365)
    location_hierarchy.add_attribute("city", 3567)
    location_hierarchy.add_attribute("street", 674339)

    location_hierarchy.generate_hierarchy()
    location_hierarchy.display_hierarchy()

    os.makedirs(output_dir, exist_ok=True)
    out_file = os.path.join(output_dir, "p6_concept_hierarchy.png")
    location_hierarchy.plot_hierarchy(out_file)
    print(f"Visualization saved to {out_file}")

    # Example 2: Product dimension
    prod_hierarchy = ConceptHierarchyGenerator("Product Dimension")
    prod_hierarchy.add_attribute("product_line", 8)
    prod_hierarchy.add_attribute("category", 42)
    prod_hierarchy.add_attribute("brand", 250)
    prod_hierarchy.add_attribute("item_sku", 18500)
    prod_hierarchy.generate_hierarchy()
    prod_hierarchy.display_hierarchy()


def main():
    output_dir = os.path.join(os.path.dirname(__file__), "images")
    solve_problem_1()
    solve_problem_2(output_dir)
    solve_problem_3()
    solve_problem_4()
    solve_problem_5(output_dir)
    solve_problem_6(output_dir)
    print("\nAll tasks in Problem Set 2 completed successfully!")


if __name__ == "__main__":
    main()
