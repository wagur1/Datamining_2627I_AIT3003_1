"""
Data Mining (AIT3003) - Problem Set 5: Density-Based Clustering (DBSCAN)
(Phân cụm dữ liệu theo mật độ)
Complete, reproducible calculations and visualizations for Questions (a) - (g).
"""

import os
from typing import Dict, List, Set, Tuple
import matplotlib.pyplot as plt
import numpy as np


def solve_dbscan(output_dir: str = "images") -> Dict[str, any]:
    print("=" * 80)
    print("PROBLEM SET 5: DENSITY-BASED CLUSTERING (DBSCAN)")
    print("=" * 80)

    # 24 Points extracted from the coordinate plane
    points: Dict[str, Tuple[int, int]] = {
        "a": (5, 8),
        "b": (10, 8),
        "c": (11, 8),
        "d": (6, 7),
        "e": (10, 7),
        "f": (12, 7),
        "g": (13, 7),
        "h": (5, 6),
        "i": (10, 6),
        "j": (13, 6),
        "k": (6, 5),
        "l": (9, 4),
        "m": (11, 5),
        "n": (14, 6),
        "o": (15, 5),
        "p": (2, 4),
        "q": (3, 4),
        "r": (5, 4),
        "s": (6, 4),
        "t": (7, 4),
        "u": (15, 4),
        "v": (3, 3),
        "w": (7, 3),
        "x": (8, 2),
    }

    eps = 2.0
    min_pts = 3

    print(f"Total points: N = {len(points)}")
    print(f"DBSCAN Parameters: eps (radius) = {eps}, MinPts = {min_pts}, Metric = Euclidean")
    print("-" * 80)

    # 1. Compute epsilon-neighborhood for each point
    neighborhoods: Dict[str, List[Tuple[str, float]]] = {}
    for name, coord in points.items():
        nbrs = []
        for other_name, other_coord in points.items():
            dist = float(np.hypot(coord[0] - other_coord[0], coord[1] - other_coord[1]))
            if dist <= eps + 1e-9:
                nbrs.append((other_name, dist))
        # Sort by distance then name
        nbrs.sort(key=lambda x: (x[1], x[0]))
        neighborhoods[name] = nbrs

    # Identify core, border, and noise points
    core_points: List[str] = []
    border_points: List[str] = []
    noise_points: List[str] = []

    print("\nEpsilon-Neighborhoods N_eps(p) and Point Classification:")
    print(f"{'Point':<6} | {'Coords':<8} | {'|N_eps|':<7} | {'Point Type':<11} | {'Neighbors within eps = 2 (with distance)'}")
    print("-" * 80)
    for name in sorted(points.keys()):
        nbr_list = neighborhoods[name]
        cnt = len(nbr_list)
        coord = points[name]
        nbr_str = ", ".join([f"{n}({d:.2f})" for n, d in nbr_list])
        if cnt >= min_pts:
            p_type = "CORE"
            core_points.append(name)
        else:
            p_type = "non-core"
        print(f"{name:<6} | {str(coord):<8} | {cnt:<7} | {p_type:<11} | {nbr_str}")

    core_set = set(core_points)

    # Distinguish border points from noise points
    for name in sorted(points.keys()):
        if name not in core_set:
            nbr_names = {n for n, _ in neighborhoods[name]}
            # If at least one neighbor is a core point, it's a border point
            if any(nbr in core_set for nbr in nbr_names):
                border_points.append(name)
            else:
                noise_points.append(name)

    print("\n" + "=" * 80)
    print("DETAILED STEP-BY-STEP ANSWERS TO QUESTIONS (a) - (g):")
    print("=" * 80)

    # (a) Core points
    print("\n(a) Danh sách các điểm lõi (Core Points, |N_eps(p)| >= 3):")
    print(f"    Tổng cộng {len(core_points)} điểm lõi:")
    print(f"    {sorted(core_points)}")
    print(f"    (4 điểm không phải điểm lõi: {border_points})")

    # (b) Directly density-reachable from d to a?
    print("\n(b) Liệu điểm a có thể đạt được mật độ trực tiếp (directly density-reachable) từ điểm d không?")
    dist_d_a = float(np.hypot(points["d"][0] - points["a"][0], points["d"][1] - points["a"][1]))
    is_d_core = "d" in core_set
    in_nbr_d = any(n == "a" for n, _ in neighborhoods["d"])
    print(f"    - Điểm d{points['d']}: |N_eps(d)| = {len(neighborhoods['d'])} >= 3 -> d là điểm lõi (CORE): {is_d_core}")
    print(f"    - Khoảng cách Euclidean dist(d, a) = sqrt((6-5)^2 + (7-8)^2) = sqrt(2) ≈ {dist_d_a:.4f} <= eps ({eps})")
    print(f"    - Do đó a in N_eps(d): {in_nbr_d}")
    print(f"    => KẾT LUẬN: CÓ (YES). Điểm a đạt được mật độ trực tiếp từ điểm d.")

    # (c) Density-reachable from i to o?
    print("\n(c) Liệu điểm o có thể đạt được mật độ (density-reachable) từ điểm i không?")
    print("    - Định nghĩa: o đạt được mật độ từ i nếu tồn tại chuỗi các điểm p1, p2, ..., pk với p1 = i, pk = o,")
    print("      trong đó tất cả các điểm p1, ..., pk-1 đều là điểm lõi và dist(pt, pt+1) <= eps.")
    print("    - Chuỗi liên kết hợp lệ từ i đến o:")
    chain_io = ["i", "e", "f", "g", "n", "o"]
    for idx in range(len(chain_io) - 1):
        u, v = chain_io[idx], chain_io[idx + 1]
        step_d = float(np.hypot(points[u][0] - points[v][0], points[u][1] - points[v][1]))
        u_is_core = u in core_set
        print(f"      Bước {idx+1}: {u}{points[u]} -> {v}{points[v]} | dist = {step_d:.4f} <= 2.0 | {u} là điểm lõi: {u_is_core}")
    print(f"    => KẾT LUẬN: CÓ (YES). Điểm o đạt được mật độ từ điểm i.")
    print(f"       Chuỗi các đối tượng trung gian: i -> e -> f -> g -> n -> o (hoặc qua 'j': i -> e -> f -> j -> n -> o).")

    # (d) Is density-reachable symmetric?
    print("\n(d) Liệu quan hệ đạt được mật độ (density-reachable) có tính chất đối xứng không?")
    print("    => KẾT LUẬN: KHÔNG ĐỐI XỨNG (Asymmetric / Not symmetric).")
    print("    - Giải thích: Nếu p là điểm lõi và q là điểm biên (border point) thuộc N_eps(p), thì q đạt được mật độ")
    print("      từ p. Tuy nhiên, vì q là điểm biên (không phải điểm lõi), không có điểm nào khác có thể đạt được")
    print("      mật độ từ q (kể cả p). Do đó p KHÔNG đạt được mật độ từ q.")
    print("    - Phản ví dụ cụ thể trong đồ thị bài tập:")
    print("      + Xét cặp điểm (w, x): w(7,3) là điểm lõi (|N_eps(w)| = 4 >= 3), x(8,2) là điểm biên (|N_eps(x)| = 2 < 3).")
    print("      + Khoảng cách dist(w, x) = sqrt(1^2 + (-1)^2) = sqrt(2) ≈ 1.414 <= 2.")
    print("      + Điểm x đạt được mật độ từ w (trực tiếp vì x in N_eps(w) và w là điểm lõi).")
    print("      + Nhưng w KHÔNG THỂ đạt được mật độ từ x vì x KHÔNG PHẢI là điểm lõi (|N_eps(x)| = 2 < 3).")
    print("      (Tương tự với các cặp: t -> l nhưng l -/-> t; i -> m nhưng m -/-> i; o -> u nhưng u -/-> o).")

    # (e) Is l density-connected with x?
    print("\n(e) Liệu điểm l có được kết nối mật độ (density-connected) với điểm x không?")
    print("    - Định nghĩa: Hai điểm p và q được kết nối mật độ nếu tồn tại một điểm lõi o' sao cho cả p và q")
    print("      đều đạt được mật độ từ o'.")
    print("    - Xét điểm l(9,4) và x(8,2):")
    print("      + Điểm l(9,4) là điểm biên của điểm lõi t(7,4) (dist(t, l) = 2.0 <= 2.0). Do đó l đạt được mật độ từ t.")
    print("      + Điểm x(8,2) là điểm biên của điểm lõi w(7,3) (dist(w, x) = sqrt(2) ≈ 1.414 <= 2.0).")
    print("      + Điểm lõi t(7,4) và điểm lõi w(7,3) nối với nhau: dist(t, w) = 1.0 <= 2.0 (cả hai đều là điểm lõi!).")
    print("      + Chọn điểm lõi chung o' = t(7,4) (hoặc o' = w(7,3)):")
    print("        * Chuỗi từ t đến l: t -> l (l đạt được mật độ từ t).")
    print("        * Chuỗi từ t đến x: t -> w -> x (x đạt được mật độ từ t, qua điểm lõi trung gian w).")
    print("    => KẾT LUẬN: CÓ (YES). Điểm l ĐƯỢC KẾT NỐI MẬT ĐỘ với điểm x.")
    print("       Điểm trung gian / điểm lõi kết nối: t(7,4) và w(7,3) (chuỗi mật độ: l <- t <-> w -> x).")

    # (f) Is density-connected symmetric?
    print("\n(f) Liệu quan hệ kết nối mật độ (density-connected) có tính chất đối xứng không?")
    print("    => KẾT LUẬN: CÓ ĐỐI XỨNG (Symmetric).")
    print("    - Giải thích: Theo định nghĩa, 'p và q kết nối mật độ' khi và chỉ khi tồn tại điểm lõi o' sao cho")
    print("      cả p và q đều đạt được mật độ từ o'. Mệnh đề này hoàn toàn đối xứng đối với p và q:")
    print("      (p và q cùng đạt được từ o') <=> (q và p cùng đạt được từ o').")
    print("      Do đó nếu p kết nối mật độ với q thì chắc chắn q cũng kết nối mật độ với p.")

    # (g) Clusters and noise
    # Standard DBSCAN cluster discovery via BFS/DFS
    visited = set()
    clusters: List[List[str]] = []

    for p in sorted(points.keys()):
        if p in visited:
            continue
        if p not in core_set:
            continue

        # Start a new cluster
        current_cluster = []
        queue = [p]
        visited.add(p)

        while queue:
            curr = queue.pop(0)
            current_cluster.append(curr)

            if curr in core_set:
                for nbr, _ in neighborhoods[curr]:
                    if nbr not in visited:
                        visited.add(nbr)
                        queue.append(nbr)

        clusters.append(sorted(current_cluster))

    # Any unvisited point is noise
    noise = [p for p in points if p not in visited]

    print("\n(g) Xác định các cụm dựa trên mật độ và các điểm nhiễu (DBSCAN Clusters & Noise):")
    print(f"    Số lượng cụm phát hiện: {len(clusters)} cụm.")
    for idx, cl in enumerate(clusters):
        c_cores = [pt for pt in cl if pt in core_set]
        c_borders = [pt for pt in cl if pt not in core_set]
        print(f"    * CỤM {idx + 1} ({len(cl)} điểm): {cl}")
        print(f"      - Điểm lõi ({len(c_cores)}): {c_cores}")
        print(f"      - Điểm biên ({len(c_borders)}): {c_borders}")
    print(f"    * ĐIỂM NHIỄU (Noise Points): {noise if noise else 'Không có (Tập rỗng ∅)'}")

    # ==========================================================================
    # VISUALIZATIONS
    # ==========================================================================
    os.makedirs(output_dir, exist_ok=True)

    # Figure 1: Full Clustering Plot
    fig, ax = plt.subplots(figsize=(12, 6.8), dpi=150)
    cluster_colors = ["#1f77b4", "#ff7f0e"]

    # Draw epsilon circles around core points
    for c_name in core_points:
        cp = points[c_name]
        circle = plt.Circle(cp, eps, color="gray", fill=False, linestyle="--", alpha=0.18, linewidth=0.8)
        ax.add_patch(circle)

    # Plot points by cluster
    for c_idx, cl in enumerate(clusters):
        col = cluster_colors[c_idx]
        for p_name in cl:
            p = points[p_name]
            is_c = p_name in core_set
            marker = "o" if is_c else "s"
            size = 140 if is_c else 100
            ax.scatter(p[0], p[1], color=col, marker=marker, s=size, edgecolors="black", linewidths=1.5, zorder=5)
            # Label
            label_txt = f"{p_name}"
            ax.annotate(label_txt, (p[0] + 0.18, p[1] + 0.18), fontsize=10, fontweight="bold", color="black", zorder=6)

    # Legend elements
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#1f77b4", markersize=10, markeredgecolor="black", label=f"Cluster 1 Core Points (11 pts)"),
        Line2D([0], [0], marker="s", color="w", markerfacecolor="#1f77b4", markersize=9, markeredgecolor="black", label=f"Cluster 1 Border Points (l, x)"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#ff7f0e", markersize=10, markeredgecolor="black", label=f"Cluster 2 Core Points (9 pts)"),
        Line2D([0], [0], marker="s", color="w", markerfacecolor="#ff7f0e", markersize=9, markeredgecolor="black", label=f"Cluster 2 Border Points (m, u)"),
        Line2D([0], [0], linestyle="--", color="gray", label=r"$\epsilon$-Neighborhood Radius ($\epsilon=2.0$)"),
    ]
    ax.legend(handles=legend_elements, loc="upper left", fontsize=9)

    ax.set_xticks(np.arange(1, 18))
    ax.set_yticks(np.arange(1, 11))
    ax.set_xlim(0.5, 17.5)
    ax.set_ylim(0.5, 10.5)
    ax.set_xlabel("X Coordinate", fontweight="bold")
    ax.set_ylabel("Y Coordinate", fontweight="bold")
    ax.set_title(r"Problem Set 5: DBSCAN Clustering Results ($\epsilon = 2.0$, $\mathrm{MinPts} = 3$)", fontweight="bold", fontsize=12)
    ax.grid(True, linestyle=":", alpha=0.6)

    plt.tight_layout()
    fig1_path = os.path.join(output_dir, "dbscan_clusters.png")
    plt.savefig(fig1_path, dpi=150)
    plt.close(fig)
    print(f"Saved DBSCAN clusters plot to: {fig1_path}")

    # Figure 2: Reachability & Connectivity Paths (Questions b, c, e)
    fig, ax = plt.subplots(figsize=(12, 6.8), dpi=150)

    # Base scatter of all points
    for p_name, p in points.items():
        is_c = p_name in core_set
        ax.scatter(p[0], p[1], color="#e0e0e0" if is_c else "#f5f5f5", marker="o" if is_c else "s", s=110, edgecolors="gray", zorder=3)
        ax.annotate(p_name, (p[0] + 0.16, p[1] + 0.16), fontsize=9, color="#666666", zorder=4)

    # (b) d -> a direct reachability arrow
    ax.annotate(
        "",
        xy=points["a"],
        xytext=points["d"],
        arrowprops=dict(arrowstyle="->", color="#9467bd", lw=2.5, shrinkA=8, shrinkB=8),
        zorder=5,
    )
    ax.scatter([points["d"][0], points["a"][0]], [points["d"][1], points["a"][1]], color="#9467bd", s=130, edgecolors="black", zorder=6)
    ax.text(5.2, 7.6, "(b) Direct Reachable\n$d \\rightarrow a$", color="#9467bd", fontweight="bold", fontsize=9)

    # (c) i -> o density-reachability path
    for k in range(len(chain_io) - 1):
        p_from = points[chain_io[k]]
        p_to = points[chain_io[k + 1]]
        ax.annotate(
            "",
            xy=p_to,
            xytext=p_from,
            arrowprops=dict(arrowstyle="->", color="#2ca02c", lw=2.2, shrinkA=8, shrinkB=8),
            zorder=5,
        )
        ax.scatter(p_from[0], p_from[1], color="#2ca02c", s=130, edgecolors="black", zorder=6)
    ax.scatter(points["o"][0], points["o"][1], color="#2ca02c", s=130, edgecolors="black", zorder=6)
    ax.text(12.5, 7.8, "(c) Path $i \\rightarrow e \\rightarrow f \\rightarrow g \\rightarrow n \\rightarrow o$", color="#2ca02c", fontweight="bold", fontsize=9.5)

    # (e) Density connectivity between l and x through t and w
    # t -> l
    ax.annotate("", xy=points["l"], xytext=points["t"], arrowprops=dict(arrowstyle="->", color="#d62728", lw=2.2, shrinkA=8, shrinkB=8), zorder=5)
    # t <-> w
    ax.annotate("", xy=points["w"], xytext=points["t"], arrowprops=dict(arrowstyle="<->", color="#d62728", lw=2.2, shrinkA=8, shrinkB=8), zorder=5)
    # w -> x
    ax.annotate("", xy=points["x"], xytext=points["w"], arrowprops=dict(arrowstyle="->", color="#d62728", lw=2.2, shrinkA=8, shrinkB=8), zorder=5)

    for pt_e in ["l", "t", "w", "x"]:
        coord_e = points[pt_e]
        ax.scatter(coord_e[0], coord_e[1], color="#d62728", s=130, marker="s" if pt_e in ["l", "x"] else "o", edgecolors="black", zorder=6)

    ax.text(7.6, 3.6, "(e) Density Connected\n$l \\leftarrow t \\leftrightarrow w \\rightarrow x$", color="#d62728", fontweight="bold", fontsize=9.5)

    ax.set_xticks(np.arange(1, 18))
    ax.set_yticks(np.arange(1, 11))
    ax.set_xlim(0.5, 17.5)
    ax.set_ylim(0.5, 10.5)
    ax.set_xlabel("X Coordinate", fontweight="bold")
    ax.set_ylabel("Y Coordinate", fontweight="bold")
    ax.set_title("Problem Set 5: Verification Paths for Questions (b), (c), and (e)", fontweight="bold", fontsize=12)
    ax.grid(True, linestyle=":", alpha=0.6)

    plt.tight_layout()
    fig2_path = os.path.join(output_dir, "dbscan_reachability_paths.png")
    plt.savefig(fig2_path, dpi=150)
    plt.close(fig)
    print(f"Saved DBSCAN paths plot to: {fig2_path}")

    return {
        "core_points": core_points,
        "border_points": border_points,
        "noise_points": noise_points,
        "clusters": clusters,
    }


if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    img_dir = os.path.join(script_dir, "images")
    solve_dbscan(output_dir=img_dir)
