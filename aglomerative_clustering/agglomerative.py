import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from collections import defaultdict

# ------------------------------------------------------------------ #
#  Agglomerative (Bottom-Up) Hierarchical Clustering                  #
# ------------------------------------------------------------------ #

class Agglomerative:
    def __init__(self, k: int = 3, linkage: str = "ward"):
        """
        Agglomerative Hierarchical Clustering from scratch.

        Starts with each point as its own cluster, then merges the
        closest pair of clusters until k clusters remain.

        Parameters:
            k       : Number of final clusters
            linkage : 'single' | 'complete' | 'average' | 'ward'
        """
        self.k = k
        self.linkage = linkage
        self.labels = None
        self.merge_history = []   # list of (cluster_a, cluster_b, distance)

    # ------------------------------------------------------------------ #
    #  Linkage distances                                                   #
    # ------------------------------------------------------------------ #

    def _cluster_distance(self, A: np.ndarray, B: np.ndarray) -> float:
        if self.linkage == "single":
            return np.min(np.linalg.norm(A[:, None] - B[None, :], axis=2))
        elif self.linkage == "complete":
            return np.max(np.linalg.norm(A[:, None] - B[None, :], axis=2))
        elif self.linkage == "average":
            return np.mean(np.linalg.norm(A[:, None] - B[None, :], axis=2))
        elif self.linkage == "ward":
            mean_A, mean_B = A.mean(axis=0), B.mean(axis=0)
            nA, nB = len(A), len(B)
            return np.sqrt(nA * nB / (nA + nB)) * np.linalg.norm(mean_A - mean_B)
        else:
            raise ValueError(f"Unknown linkage: {self.linkage}")

    # ------------------------------------------------------------------ #
    #  Fit                                                                 #
    # ------------------------------------------------------------------ #

    def fit(self, X: np.ndarray) -> "Agglomerative":
        X = np.array(X, dtype=float)
        n = len(X)

        # Each point starts as its own cluster
        clusters = {i: np.array([X[i]]) for i in range(n)}
        self.merge_history = []

        while len(clusters) > self.k:
            ids = list(clusters.keys())
            best_dist = np.inf
            best_pair = (None, None)

            # Find the closest pair of clusters
            for i in range(len(ids)):
                for j in range(i + 1, len(ids)):
                    d = self._cluster_distance(clusters[ids[i]], clusters[ids[j]])
                    if d < best_dist:
                        best_dist = d
                        best_pair = (ids[i], ids[j])

            a, b = best_pair
            self.merge_history.append((a, b, best_dist))

            # Merge b into a
            clusters[a] = np.vstack([clusters[a], clusters[b]])
            del clusters[b]

            remaining = len(clusters)
            if remaining % 20 == 0 or remaining <= self.k + 2:
                print(f"  Clusters remaining: {remaining}")

        # Assign labels
        self.labels = np.zeros(n, dtype=int)
        for label_id, (cluster_id, points) in enumerate(clusters.items()):
            for pt in points:
                idx = np.where((X == pt).all(axis=1))[0]
                if len(idx) > 0:
                    self.labels[idx[0]] = label_id

        return self

    def fit_predict(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).labels


# ------------------------------------------------------------------ #
#  Plotting                                                            #
# ------------------------------------------------------------------ #

PALETTE = ["#E63946", "#2A9D8F", "#457B9D", "#E9C46A",
           "#F4A261", "#6A4C93", "#43AA8B", "#F8961E"]


def plot_all(X: np.ndarray, model: Agglomerative, save_path: str = None):
    fig = plt.figure(figsize=(14, 5), facecolor="#0F1117")
    fig.suptitle(f"Agglomerative Clustering  (linkage='{model.linkage}', k={model.k})",
                 fontsize=14, color="white")
    gs = gridspec.GridSpec(1, 2, figure=fig, wspace=0.35)

    # ── Cluster scatter ───────────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0])
    ax1.set_facecolor("#1A1D27")
    for j in range(model.k):
        pts = X[model.labels == j]
        ax1.scatter(pts[:, 0], pts[:, 1], color=PALETTE[j % len(PALETTE)],
                    alpha=0.7, s=45, edgecolors="none", label=f"Cluster {j}")
    ax1.set_title("Cluster Assignments", color="white", fontsize=12)
    ax1.tick_params(colors="gray")
    for sp in ax1.spines.values(): sp.set_edgecolor("#333")
    ax1.legend(fontsize=8, labelcolor="white", facecolor="#1A1D27", edgecolor="#333")

    # ── Merge distances (dendrogram proxy) ───────────────────────────
    ax2 = fig.add_subplot(gs[1])
    ax2.set_facecolor("#1A1D27")
    dists = [d for _, _, d in model.merge_history]
    ax2.plot(range(1, len(dists) + 1), dists, color="#2A9D8F", linewidth=2)
    ax2.fill_between(range(1, len(dists) + 1), dists, alpha=0.15, color="#2A9D8F")
    # Mark last k-1 merges (final cluster formations)
    cutoff = len(dists) - (model.k - 1)
    ax2.axvline(x=cutoff, color="#E63946", linestyle="--", linewidth=1.5,
                label=f"Cut for k={model.k}")
    ax2.set_title("Merge Distances over Steps", color="white", fontsize=12)
    ax2.set_xlabel("Merge Step", color="gray")
    ax2.set_ylabel("Distance", color="gray")
    ax2.tick_params(colors="gray")
    ax2.legend(fontsize=9, labelcolor="white", facecolor="#1A1D27", edgecolor="#333")
    for sp in ax2.spines.values(): sp.set_edgecolor("#333")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
        print(f"  Saved → {save_path}")
    plt.show()


# ------------------------------------------------------------------ #
#  Demo                                                                #
# ------------------------------------------------------------------ #

if __name__ == "__main__":
    np.random.seed(42)
    X = np.vstack([
        np.random.randn(60, 2) * 0.8 + center
        for center in [(0, 0), (5, 1), (2.5, 5)]
    ])

    print("Fitting Agglomerative Clustering (ward linkage)...")
    model = Agglomerative(k=3, linkage="ward")
    model.fit(X)
    print(f"  Merges performed : {len(model.merge_history)}")

    plot_all(X, model, save_path="output.png")
