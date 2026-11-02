# Datamining_2627I_AIT3003_1

Repository containing coursework, problem sets, and code implementations for Data Mining (Course code: AIT3003).

## Table of Contents
- [Problem Set 1: Data Understanding](Problem%20Set%201/README.md)
  - Exercise 1: Univariate analysis on age (mean, median, mode, midrange, quartiles, five-number summary, boxplot, variance & standard deviation).
  - Exercise 2: Bivariate analysis on age and %fat (summary statistics, boxplots, scatter plot, Pearson & Spearman correlation).
  - Exercise 3: Cristiano Ronaldo Champions League dataset (summary statistics, quartiles, boxplots, pairwise scatter plots, covariance matrix).
  - Exercise 4: Contingency table and Pearson's Chi-Square test of independence (expected frequencies, test statistic, p-value, hypothesis conclusion).
- [Problem Set 2: Data Preparation](Problem%20Set%202/README.md)
  - Problem P1: Handling missing values in real-world data (deletion, manual, global constant, central tendency, class-conditional, predictive imputation, multiple imputation).
  - Problem P2: Data smoothing by bin means (bin depth 3), outlier determination (IQR, Z-score, MAD), and alternative smoothing techniques (bin medians, bin boundaries, regression, clustering).
  - Problem P3: Issues in data integration (schema integration, entity identification, redundancy & correlation analysis, tuple duplication, data value conflicts).
  - Problem P4: Mathematical value ranges of normalization methods (min-max, z-score, z-score with MAD, decimal scaling).
  - Problem P5: Normalization transformations on age value 35 (min-max onto [0, 1], z-score with std 12.94, decimal scaling) and method preference evaluation.
  - Problem P6: Automatic concept hierarchy generation algorithm for nominal data based on attribute distinct value counts (pseudocode, Python implementation, schema execution).
- [Problem Set 3: Similarity and Distance Measures](Problem%20Set%203/README.md)
  - Problem P1: Euclidean ($L_2$), Manhattan ($L_1$), and Minkowski ($L_3, q=3$) distances between numerical data objects $(22, 1, 42, 10)$ and $(20, 0, 36, 8)$.
  - Problem P2: Jaccard similarity and distance between Vietnamese word sets (“ông già đi nhanh quá” and “thời_gian trôi đi nhanh quá”).
  - Problem P3: Vocabulary construction, term-frequency (TF) vector representation, and cosine similarity between English sentences.
  - Problem P4: Levenshtein edit distance dynamic programming table and optimal edit alignment between “học sinh” and “hojc xih”.
- [Problem Set 4: Clustering](Problem%20Set%204/README.md)
  - Problem P1: Cell-based Shannon entropy $H$ for an 8-point distribution across a $4 \times 4$ spatial grid.
  - Problem P2: K-Means clustering algorithm ($k=3$, Euclidean metric) with initial seeds $A_1(2,10), B_1(5,8), C_1(1,2)$, intermediate centroids after Iteration 1, and final converged clusters.
  - Problem P3: Agglomerative hierarchical clustering on a 6-object distance matrix using Single Link (nearest neighbor) and Complete Link (furthest neighbor) with dendrograms.
- [Problem Set 5: Density-Based Clustering (DBSCAN)](Problem%20Set%205/README.md)
  - Questions (a)-(g): Complete DBSCAN evaluation ($\epsilon = 2.0, \text{MinPts} = 3$) on a 24-point 2D dataset.
  - Identification of 20 core points and 4 border points ($l, m, u, x$), verification of direct density-reachability ($d \to a$), reachability path ($i \to \dots \to o$), asymmetry of density-reachable relation, density-connectivity proof between $l$ and $x$ through $(t, w)$, symmetry of density-connected relation, and partition into 2 clusters with 0 noise points.
