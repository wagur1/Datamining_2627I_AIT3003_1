"""
Data Mining (AIT3003) - Problem Set 4: Clustering
(Bài tập 4. Phân cụm dữ liệu)
Complete, reproducible calculations and visualizations for Problems P1 - P3.
"""

import os
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
import numpy as np
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import squareform


# ==============================================================================
# PROBLEM P1: Cell-Based Entropy Calculation
# ==============================================================================
def solve_problem_1(output_dir: str = "images") -> Dict[str, float]:
    print("=" * 75)
    print("PROBLEM P1: Cell-Based Entropy for 8 Points in a 4x4 Grid")
    print("=" * 75)

    # 4x4 grid: rows 1..4 (top to bottom), cols 1..4 (left to right)
    # Cell counts based on the problem statement image:
    # Cell (row 1, col 4): 2 points
    # Cell (row 3, col 1): 4 points
    # Cell (row 4, col 3): 2 points
    # All other 13 cells: 0 points
    total_points = 8
    num_cells = 16

    grid_counts = np.zeros((4, 4), dtype=int)
    grid_counts[0, 3] = 2  # (row 1, col 4)
    grid_counts[2, 0] = 4  # (row 3, col 1)
    grid_counts[3, 2] = 2  # (row 4, col 3)

    print(f"Total points N = {total_points}")
    print(f"Total grid cells m = 4 x 4 = {num_cells}")
    print("\nPoint counts per cell in 4x4 grid:")
    print(f"{'Row \\ Col':<10} | {'Col 1':<8} | {'Col 2':<8} | {'Col 3':<8} | {'Col 4':<8}")
    print("-" * 55)
    for r in range(4):
        print(f"Row {r+1:<6} | {grid_counts[r, 0]:<8} | {grid_counts[r, 1]:<8} | {grid_counts[r, 2]:<8} | {grid_counts[r, 3]:<8}")

    # Non-empty cells
    non_empty = [
        ("Cell (Row 1, Col 4)", grid_counts[0, 3]),
        ("Cell (Row 3, Col 1)", grid_counts[2, 0]),
        ("Cell (Row 4, Col 3)", grid_counts[3, 2]),
    ]

    probabilities = [count / total_points for _, count in non_empty]

    print("\nNon-empty cell probability distribution p_i = n_i / N:")
    for (name, cnt), p in zip(non_empty, probabilities):
        print(f"  * {name}: n_i = {cnt} -> p_i = {cnt}/{total_points} = {p:.4f}")

    # Shannon Entropy with base 2
    # H = - sum(p_i * log2(p_i))
    h_base2 = -sum(p * np.log2(p) for p in probabilities)
    h_nat = -sum(p * np.log(p) for p in probabilities)
    h_normalized = h_base2 / np.log2(num_cells)  # log2(16) = 4

    print("\nCalculations:")
    print("  - Term 1 (p = 2/8 = 1/4): (1/4) * log2(1/4) = 0.25 * (-2) = -0.50")
    print("  - Term 2 (p = 4/8 = 1/2): (1/2) * log2(1/2) = 0.50 * (-1) = -0.50")
    print("  - Term 3 (p = 2/8 = 1/4): (1/4) * log2(1/4) = 0.25 * (-2) = -0.50")
    print("  - Empty cells (p = 0)  : lim_{p->0} p*log2(p) = 0 (convention)")
    print(f"\n=> Cell-based Entropy H (base 2) = -(-0.50 - 0.50 - 0.50) = {h_base2:.4f} bits")
    print(f"=> Cell-based Entropy H (natural base e) = {h_nat:.4f} nats")
    print(f"=> Normalized Entropy H / log2(16) = {h_base2:.4f} / 4.0 = {h_normalized:.4f}")

    # Visualization
    os.makedirs(output_dir, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), dpi=150)

    # Subplot 1: Grid heatmap
    im = axes[0].imshow(grid_counts, cmap="YlOrRd", vmin=0, vmax=5)
    plt.colorbar(im, ax=axes[0], fraction=0.046, pad=0.04, label="Point Count per Cell")
    axes[0].set_xticks(np.arange(4))
    axes[0].set_yticks(np.arange(4))
    axes[0].set_xticklabels([f"Col {j+1}" for j in range(4)], fontweight="bold")
    axes[0].set_yticklabels([f"Row {i+1}" for i in range(4)], fontweight="bold")

    for i in range(4):
        for j in range(4):
            cnt = grid_counts[i, j]
            txt = f"{cnt} pts\n(p={cnt/8:.2f})" if cnt > 0 else "0"
            color = "white" if cnt >= 4 else "black"
            axes[0].text(j, i, txt, ha="center", va="center", color=color, fontweight="bold", fontsize=10)

    axes[0].set_title("P1: 4x4 Grid Point Distribution ($N = 8$)", fontweight="bold")

    # Subplot 2: Entropy breakdown
    terms = [-(p * np.log2(p)) for p in probabilities]
    cell_labels = ["Cell (1,4)\n[n=2, p=1/4]", "Cell (3,1)\n[n=4, p=1/2]", "Cell (4,3)\n[n=2, p=1/4]"]
    bars = axes[1].bar(cell_labels, terms, color=["#1f77b4", "#ff7f0e", "#2ca02c"], edgecolor="black", width=0.5)
    for bar, term in zip(bars, terms):
        axes[1].text(bar.get_x() + bar.get_width() / 2, bar.get_height() / 2, f"+{term:.2f} bits", ha="center", va="center", color="white", fontweight="bold", fontsize=11)

    axes[1].axhline(0.5, color="gray", linestyle="--", alpha=0.7)
    axes[1].set_ylabel("Entropy Contribution $-p_i \\log_2(p_i)$ (bits)")
    axes[1].set_ylim(0, 0.7)
    axes[1].set_title(f"P1: Entropy Breakdown $\\sum -p_i\\log_2(p_i) = {h_base2:.1f}$ bits", fontweight="bold")
    axes[1].grid(True, axis="y", linestyle=":", alpha=0.6)

    plt.tight_layout()
    fig_path = os.path.join(output_dir, "p1_cell_entropy.png")
    plt.savefig(fig_path, dpi=150)
    plt.close(fig)
    print(f"Saved P1 plot to: {fig_path}")

    return {"h_base2": h_base2, "h_nat": h_nat, "h_normalized": h_normalized}


# ==============================================================================
# PROBLEM P2: K-Means Clustering (3 Clusters, Euclidean Distance)
# ==============================================================================
def solve_problem_2(output_dir: str = "images") -> Dict[str, any]:
    print("\n" + "=" * 75)
    print("PROBLEM P2: K-Means Clustering (k = 3)")
    print("=" * 75)

    points = {
        "A1": np.array([2.0, 10.0]),
        "A2": np.array([2.0, 5.0]),
        "A3": np.array([8.0, 4.0]),
        "B1": np.array([5.0, 8.0]),
        "B2": np.array([7.0, 5.0]),
        "B3": np.array([6.0, 4.0]),
        "C1": np.array([1.0, 2.0]),
        "C2": np.array([4.0, 9.0]),
    }

    # Initial seeds
    m1_init = points["A1"].copy()
    m2_init = points["B1"].copy()
    m3_init = points["C1"].copy()
    centroids = [m1_init, m2_init, m3_init]

    print("Given 8 points:")
    for name, pt in points.items():
        print(f"  {name}: ({int(pt[0])}, {int(pt[1])})")
    print(f"\nInitial Centroids: m1 = A1{tuple(m1_init.astype(int))}, m2 = B1{tuple(m2_init.astype(int))}, m3 = C1{tuple(m3_init.astype(int))}")
    print("-" * 75)

    iteration = 0
    prev_clusters = None
    iteration_history = []

    while True:
        iteration += 1
        clusters = {1: [], 2: [], 3: []}
        step_distances = {}

        print(f"\n--- ITERATION {iteration} ---")
        for name, pt in points.items():
            dists = [np.linalg.norm(pt - c) for c in centroids]
            chosen = int(np.argmin(dists) + 1)
            clusters[chosen].append(name)
            step_distances[name] = dists
            d_strs = ", ".join([f"d(m{i+1})={dists[i]:.4f}" for i in range(3)])
            print(f"  {name}({int(pt[0])},{int(pt[1])}): {d_strs} -> Assigned to Cluster {chosen}")

        iteration_history.append({
            "iteration": iteration,
            "centroids": [c.copy() for c in centroids],
            "clusters": {k: list(v) for k, v in clusters.items()},
        })

        print(f"\nCluster assignments after Iteration {iteration}:")
        for k in [1, 2, 3]:
            pts_str = ", ".join([f"{p}{tuple(points[p].astype(int))}" for p in clusters[k]])
            print(f"  Cluster {k} ({len(clusters[k])} pts): {{{pts_str}}}")

        # Check convergence
        if clusters == prev_clusters:
            print(f"\n>>> CONVERGENCE ACHIEVED at Iteration {iteration}! Cluster memberships unchanged.")
            break

        prev_clusters = {k: list(v) for k, v in clusters.items()}

        # Compute new centroids
        new_centroids = []
        print(f"\nUpdating Centroids after Iteration {iteration}:")
        for k in [1, 2, 3]:
            c_pts = [points[p] for p in clusters[k]]
            mean_c = np.mean(c_pts, axis=0)
            new_centroids.append(mean_c)
            coords_sum = np.sum(c_pts, axis=0)
            print(f"  m{k}^({iteration}) = sum/({len(c_pts)}) = ({coords_sum[0]:.1f}, {coords_sum[1]:.1f})/{len(c_pts)} = ({mean_c[0]:.4f}, {mean_c[1]:.4f})")

        centroids = new_centroids

    # Answers
    c_after_r1 = iteration_history[0]
    m1_r1 = np.mean([points[p] for p in c_after_r1["clusters"][1]], axis=0)
    m2_r1 = np.mean([points[p] for p in c_after_r1["clusters"][2]], axis=0)
    m3_r1 = np.mean([points[p] for p in c_after_r1["clusters"][3]], axis=0)

    print("\n" + "=" * 75)
    print("FINAL SUMMARY OF ANSWERS FOR P2:")
    print("=" * 75)
    print("a) Ba tâm cụm sau vòng thực thi đầu tiên (Centroids after Iteration 1):")
    print(f"   - m1 = ({m1_r1[0]:.4f}, {m1_r1[1]:.4f}) = (2, 10)")
    print(f"   - m2 = ({m2_r1[0]:.4f}, {m2_r1[1]:.4f}) = (6, 6)")
    print(f"   - m3 = ({m3_r1[0]:.4f}, {m3_r1[1]:.4f}) = (1.5, 3.5)")
    print("\nb) Ba cụm cuối cùng (Final 3 Clusters):")
    final_clusters = iteration_history[-1]["clusters"]
    for k in [1, 2, 3]:
        print(f"   - Cụm {k}: {final_clusters[k]}")
    final_centroids = [np.mean([points[p] for p in final_clusters[k]], axis=0) for k in [1, 2, 3]]
    print("   Final Centroids:")
    for k, fc in enumerate(final_centroids):
        print(f"   - m{k+1}* = ({fc[0]:.4f}, {fc[1]:.4f})")

    # Visualization of K-Means iterations
    os.makedirs(output_dir, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), dpi=150)
    colors = {1: "#1f77b4", 2: "#ff7f0e", 3: "#2ca02c"}

    for plot_idx, it_data in enumerate(iteration_history[:3]):
        ax = axes[plot_idx]
        it_num = it_data["iteration"]
        it_clusters = it_data["clusters"]
        it_centroids = it_data["centroids"]

        # Plot points by cluster
        for c_id in [1, 2, 3]:
            c_pts = np.array([points[p] for p in it_clusters[c_id]])
            if len(c_pts) > 0:
                ax.scatter(c_pts[:, 0], c_pts[:, 1], color=colors[c_id], s=120, edgecolors="black", zorder=4, label=f"Cluster {c_id}")
                for p_name in it_clusters[c_id]:
                    p_coord = points[p_name]
                    ax.annotate(p_name, (p_coord[0] + 0.18, p_coord[1] + 0.18), fontweight="bold", fontsize=9)

        # Plot centroids
        for c_id, cent in enumerate(it_centroids):
            ax.scatter(cent[0], cent[1], color=colors[c_id + 1], marker="X", s=220, edgecolors="black", linewidths=1.5, zorder=5, label=f"Centroid m{c_id+1}")

        ax.set_title(f"Iteration {it_num}: Cluster Assignment", fontweight="bold")
        ax.set_xlim(0, 9.5)
        ax.set_ylim(1, 11)
        ax.set_xlabel("X coordinate")
        ax.set_ylabel("Y coordinate")
        ax.grid(True, linestyle=":", alpha=0.6)
        if plot_idx == 0:
            ax.legend(loc="lower right", fontsize=8)

    plt.tight_layout()
    fig_path = os.path.join(output_dir, "p2_kmeans_iterations.png")
    plt.savefig(fig_path, dpi=150)
    plt.close(fig)
    print(f"Saved P2 plot to: {fig_path}")

    return {
        "centroids_round1": [m1_r1.tolist(), m2_r1.tolist(), m3_r1.tolist()],
        "final_clusters": final_clusters,
        "final_centroids": [fc.tolist() for fc in final_centroids],
    }


# ==============================================================================
# PROBLEM P3: Hierarchical Agglomerative Clustering (Single & Complete Link)
# ==============================================================================
def solve_problem_3(output_dir: str = "images") -> Dict[str, any]:
    print("\n" + "=" * 75)
    print("PROBLEM P3: Hierarchical Clustering (Single Link vs Complete Link)")
    print("=" * 75)

    labels = ["A", "B", "C", "D", "E", "F"]
    n = len(labels)

    # Reconstruct symmetric distance matrix
    dist_mat = np.zeros((n, n), dtype=float)
    lower_triangle = {
        (1, 0): 0.12,
        (2, 0): 0.51, (2, 1): 0.25,
        (3, 0): 0.84, (3, 1): 0.16, (3, 2): 0.14,
        (4, 0): 0.28, (4, 1): 0.77, (4, 2): 0.70, (4, 3): 0.45,
        (5, 0): 0.34, (5, 1): 0.61, (5, 2): 0.93, (5, 3): 0.20, (5, 4): 0.67,
    }
    for (i, j), val in lower_triangle.items():
        dist_mat[i, j] = val
        dist_mat[j, i] = val

    condensed = squareform(dist_mat)

    print("Given Pairwise Distance Matrix (6 objects):")
    header = f"{' ':>4} | " + " | ".join(f"{l:>6}" for l in labels)
    print(header)
    print("-" * len(header))
    for i in range(n):
        row_str = " | ".join(f"{dist_mat[i, j]:>6.2f}" for j in range(n))
        print(f"{labels[i]:>4} | {row_str}")

    # (a) Single Link Clustering
    Z_single = linkage(condensed, method="single")
    print("\n" + "-" * 50)
    print("(a) SINGLE LINK (Nearest Neighbor) Clustering Steps:")
    print("-" * 50)
    cluster_names = list(labels)
    for step_idx, row in enumerate(Z_single):
        idx1, idx2, d, size = int(row[0]), int(row[1]), row[2], int(row[3])
        new_name = f"({cluster_names[idx1]}+{cluster_names[idx2]})"
        cluster_names.append(new_name)
        print(f"Step {step_idx+1}: Merge '{cluster_names[idx1]}' and '{cluster_names[idx2]}' at dist = {d:.4f} (Cluster size: {size})")

    # (b) Complete Link Clustering
    Z_complete = linkage(condensed, method="complete")
    print("\n" + "-" * 50)
    print("(b) COMPLETE LINK (Furthest Neighbor) Clustering Steps:")
    print("-" * 50)
    cluster_names_c = list(labels)
    for step_idx, row in enumerate(Z_complete):
        idx1, idx2, d, size = int(row[0]), int(row[1]), row[2], int(row[3])
        new_name = f"({cluster_names_c[idx1]}+{cluster_names_c[idx2]})"
        cluster_names_c.append(new_name)
        print(f"Step {step_idx+1}: Merge '{cluster_names_c[idx1]}' and '{cluster_names_c[idx2]}' at dist = {d:.4f} (Cluster size: {size})")

    # Visualization: Dendrograms side by side
    os.makedirs(output_dir, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2), dpi=150)

    # Single Link Dendrogram
    axes[0].set_title("(a) Single Link Dendrogram (Min Distance)", fontweight="bold")
    dendro_s = dendrogram(Z_single, labels=labels, ax=axes[0], color_threshold=0, above_threshold_color="#1f77b4")
    axes[0].set_ylabel("Distance (Merge Height)")
    axes[0].set_xlabel("Objects")
    axes[0].grid(True, axis="y", linestyle=":", alpha=0.6)

    # Annotate heights on single link
    for i, d, c in zip(dendro_s["icoord"], dendro_s["dcoord"], dendro_s["color_list"]):
        x = 0.5 * sum(i[1:3])
        y = d[1]
        if y > 0:
            axes[0].text(x, y + 0.012, f"{y:.2f}", ha="center", va="bottom", fontsize=9, fontweight="bold", color="#1f77b4")

    # Complete Link Dendrogram
    axes[1].set_title("(b) Complete Link Dendrogram (Max Distance)", fontweight="bold")
    dendro_c = dendrogram(Z_complete, labels=labels, ax=axes[1], color_threshold=0, above_threshold_color="#d62728")
    axes[1].set_ylabel("Distance (Merge Height)")
    axes[1].set_xlabel("Objects")
    axes[1].grid(True, axis="y", linestyle=":", alpha=0.6)

    # Annotate heights on complete link
    for i, d, c in zip(dendro_c["icoord"], dendro_c["dcoord"], dendro_c["color_list"]):
        x = 0.5 * sum(i[1:3])
        y = d[1]
        if y > 0:
            axes[1].text(x, y + 0.025, f"{y:.2f}", ha="center", va="bottom", fontsize=9, fontweight="bold", color="#d62728")

    plt.tight_layout()
    fig_path = os.path.join(output_dir, "p3_dendrograms.png")
    plt.savefig(fig_path, dpi=150)
    plt.close(fig)
    print(f"Saved P3 plot to: {fig_path}")

    return {"Z_single": Z_single.tolist(), "Z_complete": Z_complete.tolist()}


if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    img_dir = os.path.join(script_dir, "images")
    solve_problem_1(output_dir=img_dir)
    solve_problem_2(output_dir=img_dir)
    solve_problem_3(output_dir=img_dir)
