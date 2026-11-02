"""
Data Mining (AIT3003) - Problem Set 3: Similarity and Distance Measures
(Bài tập 3. Các độ đo tương tự và khoảng cách)
Complete, reproducible calculations and visualizations for Problems P1 - P4.
"""

import os
import re
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
import numpy as np


# ==============================================================================
# PROBLEM P1: Euclidean, Manhattan, and Minkowski Distances
# ==============================================================================
def minkowski_distance(x1: np.ndarray, x2: np.ndarray, q: float) -> float:
    """Compute Minkowski distance of order q between two numeric vectors."""
    if np.isinf(q):
        return float(np.max(np.abs(x1 - x2)))
    return float(np.sum(np.abs(x1 - x2) ** q) ** (1.0 / q))


def solve_problem_1(output_dir: str = "images") -> Dict[str, float]:
    print("=" * 75)
    print("PROBLEM P1: Euclidean, Manhattan, and Minkowski Distances")
    print("=" * 75)

    x1 = np.array([22, 1, 42, 10], dtype=float)
    x2 = np.array([20, 0, 36, 8], dtype=float)
    diffs = np.abs(x1 - x2)

    print(f"Object 1: x1 = {tuple(x1.astype(int).tolist())}")
    print(f"Object 2: x2 = {tuple(x2.astype(int).tolist())}")
    print(f"Component-wise absolute differences |x1_k - x2_k|: {tuple(diffs.astype(int).tolist())}")
    print("-" * 75)

    # (a) Euclidean distance (q = 2)
    sq_diffs = diffs ** 2
    sum_sq = float(np.sum(sq_diffs))
    d_euclidean = minkowski_distance(x1, x2, q=2)
    print("(a) Euclidean Distance (q = 2):")
    print(f"    Squared differences: {tuple(sq_diffs.astype(int).tolist())}")
    print(f"    Sum of squares: {' + '.join(str(int(v)) for v in sq_diffs)} = {int(sum_sq)}")
    print(f"    d_Euclidean(x1, x2) = sqrt({int(sum_sq)}) = 3*sqrt(5) ≈ {d_euclidean:.4f}")

    # (b) Manhattan distance (q = 1)
    d_manhattan = minkowski_distance(x1, x2, q=1)
    print("\n(b) Manhattan (City-Block / L1) Distance (q = 1):")
    print(f"    d_Manhattan(x1, x2) = {' + '.join(str(int(v)) for v in diffs)} = {int(d_manhattan)}")

    # (c) Minkowski distance (q = 3)
    cubed_diffs = diffs ** 3
    sum_cubed = float(np.sum(cubed_diffs))
    d_minkowski_3 = minkowski_distance(x1, x2, q=3)
    print("\n(c) Minkowski Distance (q = 3):")
    print(f"    Cubed differences: {tuple(cubed_diffs.astype(int).tolist())}")
    print(f"    Sum of cubes: {' + '.join(str(int(v)) for v in cubed_diffs)} = {int(sum_cubed)}")
    print(f"    d_Minkowski_3(x1, x2) = ({int(sum_cubed)})^(1/3) ≈ {d_minkowski_3:.4f}")

    # Extra reference: Supremum / Chebyshev distance (q -> infinity)
    d_supremum = minkowski_distance(x1, x2, q=np.inf)
    print(f"\n[Reference] Supremum / Chebyshev Distance (q -> inf): max{tuple(diffs.astype(int).tolist())} = {int(d_supremum)}")

    # Visualization for P1
    os.makedirs(output_dir, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), dpi=150)

    # Subplot 1: Component-wise differences and contributions
    dims = [f"Dim {i+1}\n({int(x1[i])} vs {int(x2[i])})" for i in range(len(x1))]
    x_pos = np.arange(len(dims))
    width = 0.26

    axes[0].bar(x_pos - width, diffs, width, label=r"$|x_{1k} - x_{2k}|^1$ ($q=1$)", color="#1f77b4", edgecolor="black")
    axes[0].bar(x_pos, sq_diffs, width, label=r"$|x_{1k} - x_{2k}|^2$ ($q=2$)", color="#ff7f0e", edgecolor="black")
    axes[0].bar(x_pos + width, cubed_diffs, width, label=r"$|x_{1k} - x_{2k}|^3$ ($q=3$)", color="#2ca02c", edgecolor="black")
    axes[0].set_yscale("symlog", linthresh=1)
    axes[0].set_xticks(x_pos)
    axes[0].set_xticklabels(dims)
    axes[0].set_ylabel("Power of Absolute Difference (Log Scale)")
    axes[0].set_title("P1: Component Contributions by Order $q$", fontweight="bold")
    axes[0].grid(True, linestyle=":", alpha=0.6)
    axes[0].legend()

    # Subplot 2: Minkowski distance curve L_q as a function of q
    q_vals = np.linspace(1.0, 10.0, 200)
    d_vals = [minkowski_distance(x1, x2, q) for q in q_vals]
    axes[1].plot(q_vals, d_vals, color="#1f77b4", linewidth=2.5, label=r"Minkowski $L_q(x_1, x_2)$")
    axes[1].axhline(d_supremum, color="gray", linestyle="--", linewidth=1.5, label=rf"Supremum Limit $L_\infty = {int(d_supremum)}$")

    # Highlight q = 1, 2, 3
    highlight_q = [(1, d_manhattan, "Manhattan ($q=1$)", "#d62728", (1.3, 10.7)),
                   (2, d_euclidean, "Euclidean ($q=2$)", "#ff7f0e", (2.4, 7.3)),
                   (3, d_minkowski_3, "Minkowski ($q=3$)", "#2ca02c", (3.5, 6.55))]
    for q_pt, d_pt, label_txt, col, text_pos in highlight_q:
        axes[1].scatter([q_pt], [d_pt], color=col, s=70, zorder=5, edgecolors="black")
        axes[1].annotate(
            f"{label_txt}\n$d = {d_pt:.4f}$",
            xy=(q_pt, d_pt),
            xytext=text_pos,
            arrowprops=dict(arrowstyle="->", color=col, lw=1.2),
            fontsize=9,
            fontweight="bold",
            color=col,
        )

    axes[1].set_xlabel("Minkowski Order $q$")
    axes[1].set_ylabel("Distance $d(x_1, x_2)$")
    axes[1].set_title(r"P1: Minkowski Distance Decay as $q \to \infty$", fontweight="bold")
    axes[1].set_ylim(5.5, 11.8)
    axes[1].grid(True, linestyle=":", alpha=0.6)
    axes[1].legend(loc="upper right")

    plt.tight_layout()
    fig_path = os.path.join(output_dir, "p1_minkowski_distances.png")
    plt.savefig(fig_path, dpi=150)
    plt.close(fig)
    print(f"Saved P1 plot to: {fig_path}")

    return {
        "euclidean": d_euclidean,
        "manhattan": d_manhattan,
        "minkowski_q3": d_minkowski_3,
        "supremum": d_supremum,
    }


# ==============================================================================
# PROBLEM P2: Jaccard Similarity Between Two Vietnamese Sentences
# ==============================================================================
def solve_problem_2(output_dir: str = "images") -> Dict[str, float]:
    print("\n" + "=" * 75)
    print("PROBLEM P2: Jaccard Similarity Between Word Sets")
    print("=" * 75)

    sent_a = "ông già đi nhanh quá"
    sent_b = "thời_gian trôi đi nhanh quá"

    # Case 1: Standard whitespace tokenization (matching explicit underscore in 'thời_gian')
    tokens_a = sent_a.split()
    tokens_b = sent_b.split()

    set_a = set(tokens_a)
    set_b = set(tokens_b)

    intersection = sorted(set_a.intersection(set_b), key=lambda w: (tokens_a + tokens_b).index(w))
    union = sorted(set_a.union(set_b), key=lambda w: (tokens_a + tokens_b).index(w))
    only_a = [w for w in tokens_a if w not in set_b]
    only_b = [w for w in tokens_b if w not in set_a]

    jaccard_sim = len(intersection) / len(union)
    jaccard_dist = 1.0 - jaccard_sim

    print(f"Sentence (a): \"{sent_a}\"")
    print(f"Sentence (b): \"{sent_b}\"")
    print("-" * 75)
    print("1. Primary Analysis (Whitespace Tokenization as Given in Prompt):")
    print(f"   Set A (|A| = {len(set_a)}): {tokens_a}")
    print(f"   Set B (|B| = {len(set_b)}): {tokens_b}")
    print(f"   Intersection A ∩ B (|A ∩ B| = {len(intersection)}): {intersection}")
    print(f"   Union A ∪ B (|A ∪ B| = {len(union)}): {union}")
    print(f"   Jaccard Similarity J(A, B) = |A ∩ B| / |A ∪ B| = {len(intersection)} / {len(union)} ≈ {jaccard_sim:.4f}")
    print(f"   Jaccard Distance d_J(A, B) = 1 - J(A, B) = {len(union) - len(intersection)} / {len(union)} ≈ {jaccard_dist:.4f}")

    # Case 2: Compound-word segmented variant ('ông_già' as 1 compound word)
    tokens_a_compound = ["ông_già", "đi", "nhanh", "quá"]
    set_a_comp = set(tokens_a_compound)
    inter_comp = set_a_comp.intersection(set_b)
    union_comp = set_a_comp.union(set_b)
    jaccard_sim_comp = len(inter_comp) / len(union_comp)

    print("\n2. Alternative Note (If 'ông già' is also segmented as compound word 'ông_già'):")
    print(f"   Set A' (|A'| = {len(set_a_comp)}): {tokens_a_compound}")
    print(f"   Set B  (|B|  = {len(set_b)}): {tokens_b}")
    print(f"   |A' ∩ B| = {len(inter_comp)}, |A' ∪ B| = {len(union_comp)}")
    print(f"   Jaccard Similarity J(A', B) = {len(inter_comp)} / {len(union_comp)} = {jaccard_sim_comp:.4f}")

    # Visualization for P2
    os.makedirs(output_dir, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), dpi=150)

    # Subplot 1: Binary presence/absence heatmap across the union vocabulary
    presence_matrix = np.array([
        [1 if w in set_a else 0 for w in union],
        [1 if w in set_b else 0 for w in union],
    ])
    im = axes[0].imshow(presence_matrix, cmap="Blues", aspect="auto", vmin=0, vmax=1.2)
    axes[0].set_xticks(np.arange(len(union)))
    axes[0].set_xticklabels(union, fontsize=10, fontweight="bold")
    axes[0].set_yticks([0, 1])
    axes[0].set_yticklabels(["Sentence (a)", "Sentence (b)"], fontsize=10, fontweight="bold")
    axes[0].set_title("P2: Binary Set Membership Across Union Vocabulary", fontweight="bold")

    for r in range(2):
        for c in range(len(union)):
            val = presence_matrix[r, c]
            is_shared = union[c] in intersection
            txt_color = "white" if val == 1 else "#555555"
            tag = "1\n(Shared)" if (val == 1 and is_shared) else str(val)
            axes[0].text(c, r, tag, ha="center", va="center", color=txt_color, fontsize=9, fontweight="bold")

    # Subplot 2: Set Breakdown Bar Chart
    categories = ["Only in A\n(|A \\ B|)", "Intersection\n(|A ∩ B|)", "Only in B\n(|B \\ A|)", "Total Union\n(|A ∪ B|)"]
    counts = [len(only_a), len(intersection), len(only_b), len(union)]
    word_labels = ["\n".join(only_a), "\n".join(intersection), "\n".join(only_b), f"J(A,B) = {len(intersection)}/{len(union)}\n≈ {jaccard_sim:.4f}"]
    colors = ["#4c78a8", "#2ca02c", "#f58518", "#72b7b2"]

    bars = axes[1].bar(categories, counts, color=colors, edgecolor="black", width=0.55)
    for bar, cnt, w_lbl in zip(bars, counts, word_labels):
        axes[1].text(
            bar.get_x() + bar.get_width() / 2.0,
            bar.get_height() / 2.0,
            f"Count = {cnt}\n({w_lbl})",
            ha="center",
            va="center",
            fontsize=8.5,
            fontweight="bold",
            color="white" if cnt <= 3 else "black",
        )
    axes[1].set_ylabel("Number of Distinct Words")
    axes[1].set_ylim(0, 8.2)
    axes[1].set_title(f"P2: Jaccard Similarity = {len(intersection)}/{len(union)} ≈ {jaccard_sim:.4f}", fontweight="bold")
    axes[1].grid(True, axis="y", linestyle=":", alpha=0.6)

    plt.tight_layout()
    fig_path = os.path.join(output_dir, "p2_jaccard_similarity.png")
    plt.savefig(fig_path, dpi=150)
    plt.close(fig)
    print(f"Saved P2 plot to: {fig_path}")

    return {
        "jaccard_similarity": jaccard_sim,
        "jaccard_distance": jaccard_dist,
        "jaccard_similarity_compound": jaccard_sim_comp,
    }


# ==============================================================================
# PROBLEM P3: Term-Frequency Vectors and Cosine Similarity
# ==============================================================================
def tokenize_sentence(sentence: str, lowercase: bool = True) -> List[str]:
    cleaned = re.sub(r"[^\w\s]", "", sentence)
    if lowercase:
        cleaned = cleaned.lower()
    return cleaned.split()


def compute_cosine_similarity(vec_a: np.ndarray, vec_b: np.ndarray) -> Tuple[float, float, float, float]:
    dot_prod = float(np.dot(vec_a, vec_b))
    norm_a = float(np.linalg.norm(vec_a))
    norm_b = float(np.linalg.norm(vec_b))
    cos_sim = dot_prod / (norm_a * norm_b)
    return cos_sim, dot_prod, norm_a, norm_b


def solve_problem_3(output_dir: str = "images") -> Dict[str, float]:
    print("\n" + "=" * 75)
    print("PROBLEM P3: Term Frequency (TF) Vectors and Cosine Similarity")
    print("=" * 75)

    doc_a = "The sly fox jumped over the lazy dog."
    doc_b = "The dog jumped at the intruder."

    print(f"Sentence (a): \"{doc_a}\"")
    print(f"Sentence (b): \"{doc_b}\"")
    print("-" * 75)

    # 1. Standard Case-Insensitive (Lowercased) Analysis
    tokens_a = tokenize_sentence(doc_a, lowercase=True)
    tokens_b = tokenize_sentence(doc_b, lowercase=True)

    # Preserve order of first appearance for clean presentation
    vocab: List[str] = []
    for w in tokens_a + tokens_b:
        if w not in vocab:
            vocab.append(w)

    tf_a = np.array([tokens_a.count(w) for w in vocab], dtype=int)
    tf_b = np.array([tokens_b.count(w) for w in vocab], dtype=int)

    cos_sim, dot_ab, norm_a, norm_b = compute_cosine_similarity(tf_a, tf_b)
    angle_deg = float(np.degrees(np.arccos(np.clip(cos_sim, -1.0, 1.0))))

    print("1. Primary Analysis (Standard Case-Insensitive / Lowercased Vocabulary):")
    print(f"   Tokens (a) ({len(tokens_a)} words): {tokens_a}")
    print(f"   Tokens (b) ({len(tokens_b)} words): {tokens_b}")
    print(f"   Vocabulary V ({len(vocab)} unique words): {vocab}\n")

    print(f"   {'Word':<12} | {'TF in (a)':<10} | {'TF in (b)':<10} | {'Product (d_a * d_b)':<20}")
    print("   " + "-" * 58)
    for w, ca, cb in zip(vocab, tf_a, tf_b):
        print(f"   {w:<12} | {ca:<10} | {cb:<10} | {ca * cb:<20}")
    print("   " + "-" * 58)

    print(f"\n   Vector d_a = {tuple(tf_a.tolist())}")
    print(f"   Vector d_b = {tuple(tf_b.tolist())}")
    print(f"   Dot Product (d_a · d_b) = {int(dot_ab)}")
    print(f"   Norm ||d_a|| = sqrt({int(np.sum(tf_a**2))}) ≈ {norm_a:.4f}")
    print(f"   Norm ||d_b|| = sqrt({int(np.sum(tf_b**2))}) = 2*sqrt(2) ≈ {norm_b:.4f}")
    print(f"   Cosine Similarity = {int(dot_ab)} / (sqrt(10) * sqrt(8)) = 6 / sqrt(80) = 3 / (2*sqrt(5)) ≈ {cos_sim:.4f}")
    print(f"   Angular separation θ = arccos({cos_sim:.4f}) ≈ {angle_deg:.2f}°")

    # 2. Alternative Case-Sensitive Analysis ('The' vs 'the' distinct)
    tokens_a_cs = tokenize_sentence(doc_a, lowercase=False)
    tokens_b_cs = tokenize_sentence(doc_b, lowercase=False)
    vocab_cs: List[str] = []
    for w in tokens_a_cs + tokens_b_cs:
        if w not in vocab_cs:
            vocab_cs.append(w)

    tf_a_cs = np.array([tokens_a_cs.count(w) for w in vocab_cs], dtype=int)
    tf_b_cs = np.array([tokens_b_cs.count(w) for w in vocab_cs], dtype=int)
    cos_sim_cs, dot_cs, norm_a_cs, norm_b_cs = compute_cosine_similarity(tf_a_cs, tf_b_cs)

    print("\n2. Alternative Note (Case-Sensitive where 'The' != 'the'):")
    print(f"   Vocabulary V_cs ({len(vocab_cs)} words): {vocab_cs}")
    print(f"   Vector d_a_cs = {tuple(tf_a_cs.tolist())}, ||d_a_cs|| = sqrt(8) ≈ {norm_a_cs:.4f}")
    print(f"   Vector d_b_cs = {tuple(tf_b_cs.tolist())}, ||d_b_cs|| = sqrt(6) ≈ {norm_b_cs:.4f}")
    print(f"   Dot Product = {int(dot_cs)}, Cosine Similarity = 4 / sqrt(48) = 1 / sqrt(3) ≈ {cos_sim_cs:.4f}")

    # Visualization for P3
    os.makedirs(output_dir, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8), dpi=150)

    # Subplot 1: Term Frequency Comparison Bar Chart
    x_idx = np.arange(len(vocab))
    width = 0.36
    axes[0].bar(x_idx - width / 2, tf_a, width, label="Sentence (a) TF Vector $\\mathbf{d}_a$", color="#1f77b4", edgecolor="black")
    axes[0].bar(x_idx + width / 2, tf_b, width, label="Sentence (b) TF Vector $\\mathbf{d}_b$", color="#ff7f0e", edgecolor="black")
    axes[0].set_xticks(x_idx)
    axes[0].set_xticklabels(vocab, rotation=25, fontweight="bold")
    axes[0].set_yticks([0, 1, 2])
    axes[0].set_ylim(0, 2.5)
    axes[0].set_ylabel("Term Frequency (TF)")
    axes[0].set_title("P3: Term-Frequency Vectors Across 9-Word Vocabulary", fontweight="bold")
    axes[0].grid(True, axis="y", linestyle=":", alpha=0.6)
    axes[0].legend()

    # Highlight shared words with annotations
    for idx, w in enumerate(vocab):
        if tf_a[idx] > 0 and tf_b[idx] > 0:
            axes[0].text(
                idx,
                max(tf_a[idx], tf_b[idx]) + 0.12,
                f"prod={tf_a[idx]*tf_b[idx]}",
                ha="center",
                va="bottom",
                fontsize=8.5,
                fontweight="bold",
                color="#2ca02c",
            )

    # Subplot 2: 2D Geometric Representation of Angle Between Vectors d_a and d_b
    theta_rad = np.arccos(cos_sim)
    va_2d = np.array([norm_a, 0.0])
    vb_2d = np.array([norm_b * np.cos(theta_rad), norm_b * np.sin(theta_rad)])

    axes[1].quiver(0, 0, va_2d[0], va_2d[1], angles="xy", scale_units="xy", scale=1, color="#1f77b4", width=0.015,
                   label=rf"$\mathbf{{d}}_a$ ($\|\mathbf{{d}}_a\| = \sqrt{{10}} \approx {norm_a:.2f}$)")
    axes[1].quiver(0, 0, vb_2d[0], vb_2d[1], angles="xy", scale_units="xy", scale=1, color="#ff7f0e", width=0.015,
                   label=rf"$\mathbf{{d}}_b$ ($\|\mathbf{{d}}_b\| = \sqrt{{8}} \approx {norm_b:.2f}$)")

    # Draw projection dashed line
    axes[1].plot([vb_2d[0], vb_2d[0]], [0, vb_2d[1]], linestyle="--", color="gray", linewidth=1.3, label="Orthogonal Projection")

    # Draw angle arc
    arc_angles = np.linspace(0, theta_rad, 60)
    arc_r = 1.15
    axes[1].plot(arc_r * np.cos(arc_angles), arc_r * np.sin(arc_angles), color="#d62728", linewidth=2)
    axes[1].text(
        1.32 * np.cos(theta_rad / 2),
        1.32 * np.sin(theta_rad / 2),
        f"$\\theta \\approx {angle_deg:.2f}^\\circ$\n$\\cos(\\theta) \\approx {cos_sim:.4f}$",
        fontsize=9.5,
        fontweight="bold",
        color="#d62728",
    )

    axes[1].set_xlim(-0.3, 3.6)
    axes[1].set_ylim(-0.3, 2.7)
    axes[1].set_aspect("equal", adjustable="box")
    axes[1].set_xlabel("Direction of $\\mathbf{d}_a$")
    axes[1].set_ylabel("Orthogonal Component in $\\mathrm{span}(\\mathbf{d}_a, \\mathbf{d}_b)$")
    axes[1].set_title(f"P3: Cosine Similarity = $6/\\sqrt{{80}} \\approx {cos_sim:.4f}$", fontweight="bold")
    axes[1].grid(True, linestyle=":", alpha=0.6)
    axes[1].legend(loc="upper right", fontsize=8.5)

    plt.tight_layout()
    fig_path = os.path.join(output_dir, "p3_cosine_similarity.png")
    plt.savefig(fig_path, dpi=150)
    plt.close(fig)
    print(f"Saved P3 plot to: {fig_path}")

    return {
        "cosine_similarity": cos_sim,
        "dot_product": dot_ab,
        "norm_a": norm_a,
        "norm_b": norm_b,
        "cosine_similarity_case_sensitive": cos_sim_cs,
    }


# ==============================================================================
# PROBLEM P4: Levenshtein Edit Distance Dynamic Programming Table
# ==============================================================================
def compute_levenshtein_dp(
    s1: str, s2: str, sub_cost: int = 1
) -> Tuple[np.ndarray, List[Tuple[int, int, str]]]:
    """
    Compute the Levenshtein edit distance DP matrix and optimal backtrace path
    between source string s1 and target string s2.
    Operations: Delete (cost 1), Insert (cost 1), Substitute (cost sub_cost if chars differ, 0 if match).
    """
    m, n = len(s1), len(s2)
    dp = np.zeros((m + 1, n + 1), dtype=int)

    for i in range(m + 1):
        dp[i, 0] = i
    for j in range(n + 1):
        dp[0, j] = j

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            cost = 0 if s1[i - 1] == s2[j - 1] else sub_cost
            dp[i, j] = min(
                dp[i - 1, j] + 1,         # Deletion
                dp[i, j - 1] + 1,         # Insertion
                dp[i - 1, j - 1] + cost,  # Match or Substitution
            )

    # Backtrace to recover an optimal edit sequence
    path: List[Tuple[int, int, str]] = []
    i, j = m, n
    while i > 0 or j > 0:
        if i > 0 and j > 0 and s1[i - 1] == s2[j - 1] and dp[i, j] == dp[i - 1, j - 1]:
            path.append((i, j, f"Match '{s1[i-1]}'"))
            i -= 1
            j -= 1
        elif j > 0 and dp[i, j] == dp[i, j - 1] + 1:
            path.append((i, j, f"Insert '{s2[j-1]}'"))
            j -= 1
        elif i > 0 and dp[i, j] == dp[i - 1, j] + 1:
            path.append((i, j, f"Delete '{s1[i-1]}'"))
            i -= 1
        else:
            path.append((i, j, f"Sub '{s1[i-1]}' -> '{s2[j-1]}'"))
            i -= 1
            j -= 1

    path.append((0, 0, "Start"))
    path.reverse()
    return dp, path


def format_char(ch: str) -> str:
    return "'_'" if ch == " " else f"'{ch}'"


def solve_problem_4(output_dir: str = "images") -> Dict[str, int]:
    print("\n" + "=" * 75)
    print("PROBLEM P4: Levenshtein Edit Distance Dynamic Programming Table")
    print("=" * 75)

    s1 = "học sinh"
    s2 = "hojc xih"

    print(f"Source string s1 (length m = {len(s1)}): \"{s1}\" -> Characters: {[c for c in s1]}")
    print(f"Target string s2 (length n = {len(s2)}): \"{s2}\" -> Characters: {[c for c in s2]}")
    print("Note: 'ọ' is treated as 1 single character, and space ' ' (shown as '_') is 1 character.")
    print("-" * 75)

    dp, path = compute_levenshtein_dp(s1, s2, sub_cost=1)
    edit_dist = int(dp[len(s1), len(s2)])

    # Print formatted DP matrix
    col_headers = ["ε"] + [format_char(c) for c in s2]
    row_headers = ["ε"] + [format_char(c) for c in s1]

    print("1. Levenshtein DP Table (Costs: Delete = 1, Insert = 1, Substitute = 1):\n")
    header_line = f"{'s1 \\ s2':<8} | " + " | ".join(f"{h:>5}" for h in col_headers)
    print("   " + header_line)
    print("   " + "-" * len(header_line))
    for i, r_lbl in enumerate(row_headers):
        row_str = " | ".join(f"{dp[i, j]:>5}" for j in range(len(col_headers)))
        print(f"   {r_lbl:<8} | {row_str}")

    print(f"\n   Minimum Levenshtein Edit Distance d_L(\"{s1}\", \"{s2}\") = {edit_dist}")

    print("\n2. Optimal Edit Sequence (Step-by-Step Alignment):")
    for step_idx, (r, c, op) in enumerate(path):
        print(f"   Step {step_idx:>2}: Cell ({r}, {c}) [cost = {dp[r, c]}] -> {op}")

    # Also check without whitespace & with sub_cost = 2
    s1_nospace = s1.replace(" ", "")
    s2_nospace = s2.replace(" ", "")
    dp_nospace, _ = compute_levenshtein_dp(s1_nospace, s2_nospace, sub_cost=1)
    dp_sub2, _ = compute_levenshtein_dp(s1, s2, sub_cost=2)

    print("\n3. Additional Verification Variants:")
    print(f"   - Without space character (\"{s1_nospace}\" vs \"{s2_nospace}\", 7x7): Distance = {int(dp_nospace[-1, -1])}")
    print(f"   - With Substitution Cost = 2 (Jurafsky & Martin convention): Distance = {int(dp_sub2[-1, -1])}")

    # Visualization for P4
    os.makedirs(output_dir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8.8, 7.2), dpi=150)

    im = ax.imshow(dp, cmap="YlGnBu", aspect="equal")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Cumulative Edit Cost")

    ax.set_xticks(np.arange(len(col_headers)))
    ax.set_yticks(np.arange(len(row_headers)))
    ax.set_xticklabels([f"{ j }\n{col_headers[j]}" for j in range(len(col_headers))], fontsize=10, fontweight="bold")
    ax.set_yticklabels([f"{ i }: {row_headers[i]}" for i in range(len(row_headers))], fontsize=10, fontweight="bold")
    ax.xaxis.tick_top()
    ax.xaxis.set_label_position("top")

    # Highlight backtrace cells
    path_coords = {(r, c) for r, c, _ in path}
    for i in range(dp.shape[0]):
        for j in range(dp.shape[1]):
            in_path = (i, j) in path_coords
            val = dp[i, j]
            text_color = "#d62728" if in_path else ("white" if val > 4 else "black")
            bbox_props = (
                dict(boxstyle="round,pad=0.25", facecolor="#fff3cd", edgecolor="#d62728", lw=2.0)
                if in_path
                else None
            )
            ax.text(
                j,
                i,
                str(val),
                ha="center",
                va="center",
                color=text_color,
                fontsize=11 if in_path else 10,
                fontweight="bold" if in_path else "normal",
                bbox=bbox_props,
            )

    # Draw arrows along the optimal backtrace path
    for k in range(len(path) - 1):
        r1, c1, _ = path[k]
        r2, c2, op_txt = path[k + 1]
        arrow_col = "#2ca02c" if op_txt.startswith("Match") else "#d62728"
        ax.annotate(
            "",
            xy=(c2, r2),
            xytext=(c1, r1),
            arrowprops=dict(arrowstyle="->", color=arrow_col, lw=2.2, shrinkA=12, shrinkB=12),
        )

    ax.set_xlabel("Target String $s_2$: \"hojc xih\" (Columns $j = 0 \\dots 8$)", fontsize=11, fontweight="bold", labelpad=10)
    ax.set_ylabel("Source String $s_1$: \"học sinh\" (Rows $i = 0 \\dots 8$)", fontsize=11, fontweight="bold")
    ax.set_title(
        f"P4: Levenshtein Edit Distance DP Table (\"học sinh\" $\\rightarrow$ \"hojc xih\", Distance = {edit_dist})",
        fontsize=12,
        fontweight="bold",
        pad=22,
    )

    plt.tight_layout()
    fig_path = os.path.join(output_dir, "p4_levenshtein_dp_matrix.png")
    plt.savefig(fig_path, dpi=150)
    plt.close(fig)
    print(f"Saved P4 plot to: {fig_path}")

    return {
        "levenshtein_distance": edit_dist,
        "levenshtein_nospace": int(dp_nospace[-1, -1]),
        "levenshtein_sub2": int(dp_sub2[-1, -1]),
    }


if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    img_dir = os.path.join(script_dir, "images")
    solve_problem_1(output_dir=img_dir)
    solve_problem_2(output_dir=img_dir)
    solve_problem_3(output_dir=img_dir)
    solve_problem_4(output_dir=img_dir)
