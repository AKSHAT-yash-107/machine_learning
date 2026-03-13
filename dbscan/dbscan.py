import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches

# ------------------------------------------------------------------ #
#  DBSCAN — Density-Based Spatial Clustering of Applications          #
#           with Noise                                                 #
# ------------------------------------------------------------------ #

NOISE = -1

class DBSCAN:
    def __init__(self, eps: float = 0.5, min_samples: int = 5):
        """
        DBSCAN clustering from scratch.

        Discovers clusters of arbitrary shape and labels outliers as noise.

        Parameters:
            eps        : Neighbourhood radius
            min_samples: Minimum points to form a core point
        """
        self.eps = eps
        self.min_samples = min_samples
        self.labels = None
        self.n_clusters = None
        self.n_noise = None
        self.core_mask = None

    # ------------------------------------------------------------------ #
    #  Core helpers                                                        #
    # ------------------------------------------------------------------ #

    def _region_query(self, X: np.ndarray, idx: int) -> list:
        """Return indices of all points within eps of point idx."""
        dists = np.linalg.norm(X - X[idx], axis=1)
        return list(np.where(dists <= self.eps)[0])

    def _expand_cluster(self, X: np.ndarray, labels: np.ndarray,
                         idx: int, neighbours: list, cluster_id: int) -> None:
        """Expand cluster from core point idx via BFS."""
        labels[idx] = cluster_id
        queue = list(neighbours)

        while queue:
            pt = queue.pop(0)
            if labels[pt] == NOISE:
                labels[pt] = cluster_id          # border point
            if labels[pt] != -2:                  # already visited
                continue
            labels[pt] = cluster_id
            new_neighbours = self._region_query(X, pt)
            if len(new_neighbours) >= self.min_samples:
                queue.extend(new_neighbours)

    # ------------------------------------------------------------------ #
    #  Fit                                                                 #
    # ------------------------------------------------------------------ #

    def fit(self, X: np.ndarray) -> "DBSCAN":
        X = np.array(X, dtype=float)
        n = len(X)
        labels = np.full(n, -2, dtype=int)   # -2 = unvisited
        cluster_id = 0

        for i in range(n):
            if labels[i] != -2:
                continue
            neighbours = self._region_query(X, i)
            if len(neighbours) < self.min_samples:
                labels[i] = NOISE
            else:
                self._expand_cluster(X, labels, i, neighbours, cluster_id)
                cluster_id += 1

        # Anything still -2 becomes noise (shouldn't happen but safety net)
        labels[labels == -2] = NOISE

        self.labels = labels
        self.n_clusters = cluster_id
        self.n_noise = int((labels == NOISE).sum())

        # Mark core points
        self.core_mask = np.array([
            len(self._region_query(X, i)) >= self.min_samples
            for i in range(n)
        ])

        print(f"  Clusters found : {self.n_clusters}")
        print(f"  Noise points   : {self.n_noise}")
        return self

    def fit_predict(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).labels


# ------------------------------------------------------------------ #
#  Plotting                                                            #
# ------------------------------------------------------------------ #

PALETTE = ["#E63946", "#2A9D8F", "#457B9D", "#E9C46A",
           "#F4A261", "#6A4C93", "#43AA8B", "#F8961E"]


def plot_all(X: np.ndarray, model: DBSCAN, save_path: str = None):
    fig = plt.figure(figsize=(16, 5), facecolor="#0F1117")
    fig.suptitle(
        f"DBSCAN  (eps={model.eps}, min_samples={model.min_samples})  "
        f"→  {model.n_clusters} clusters, {model.n_noise} noise pts",
        fontsize=13, color="white"
    )
    gs = gridspec.GridSpec(1, 3, figure=fig, wspace=0.35)

    unique_labels = sorted(set(model.labels))

    # ── Panel 1: Cluster assignments ─────────────────────────────────
    ax1 = fig.add_subplot(gs[0])
    ax1.set_facecolor("#1A1D27")
    for lab in unique_labels:
        mask = model.labels == lab
        color = "#555555" if lab == NOISE else PALETTE[lab % len(PALETTE)]
        label_str = "Noise" if lab == NOISE else f"Cluster {lab}"
        ax1.scatter(X[mask, 0], X[mask, 1], c=color, s=35,
                    alpha=0.8, edgecolors="none", label=label_str)
    ax1.set_title("Cluster Assignments", color="white", fontsize=11)
    ax1.tick_params(colors="gray")
    for sp in ax1.spines.values(): sp.set_edgecolor("#333")
    ax1.legend(fontsize=7, labelcolor="white", facecolor="#1A1D27",
               edgecolor="#333", ncol=2)

    # ── Panel 2: Core vs border vs noise ─────────────────────────────
    ax2 = fig.add_subplot(gs[1])
    ax2.set_facecolor("#1A1D27")
    noise_mask = model.labels == NOISE
    border_mask = (~noise_mask) & (~model.core_mask)
    core_mask = model.core_mask

    ax2.scatter(X[noise_mask, 0], X[noise_mask, 1], c="#555555",
                s=25, alpha=0.6, label="Noise")
    ax2.scatter(X[border_mask, 0], X[border_mask, 1], c="#E9C46A",
                s=35, alpha=0.7, label="Border")
    ax2.scatter(X[core_mask, 0], X[core_mask, 1], c="#E63946",
                s=55, alpha=0.8, label="Core")
    ax2.set_title("Point Types", color="white", fontsize=11)
    ax2.tick_params(colors="gray")
    for sp in ax2.spines.values(): sp.set_edgecolor("#333")
    ax2.legend(fontsize=8, labelcolor="white", facecolor="#1A1D27", edgecolor="#333")

    # ── Panel 3: Cluster size distribution ───────────────────────────
    ax3 = fig.add_subplot(gs[2])
    ax3.set_facecolor("#1A1D27")
    cluster_labels = [l for l in unique_labels if l != NOISE]
    sizes = [int((model.labels == l).sum()) for l in cluster_labels]
    colors = [PALETTE[l % len(PALETTE)] for l in cluster_labels]
    bars = ax3.bar([f"C{l}" for l in cluster_labels], sizes, color=colors, alpha=0.85)
    for bar, sz in zip(bars, sizes):
        ax3.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                 str(sz), ha="center", va="bottom", color="white", fontsize=9)
    ax3.set_title("Cluster Sizes", color="white", fontsize=11)
    ax3.set_ylabel("Points", color="gray")
    ax3.tick_params(colors="gray")
    for sp in ax3.spines.values(): sp.set_edgecolor("#333")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
        print(f"  Saved → {save_path}")
    plt.show()


# ------------------------------------------------------------------ #
#  Demo — rings + blobs to show arbitrary-shape advantage             #
# ------------------------------------------------------------------ #

if __name__ == "__main__":
    np.random.seed(42)

    # Two concentric rings
    def make_ring(n, r, noise=0.15):
        angles = np.random.uniform(0, 2 * np.pi, n)
        return np.column_stack([
            r * np.cos(angles) + np.random.randn(n) * noise,
            r * np.sin(angles) + np.random.randn(n) * noise,
        ])

    ring1 = make_ring(150, 1.0)
    ring2 = make_ring(150, 2.2)
    blob = np.random.randn(80, 2) * 0.4 + [5, 0]
    noise_pts = np.random.uniform(-4, 7, (20, 2))
    X = np.vstack([ring1, ring2, blob, noise_pts])

    print("Fitting DBSCAN...")
    model = DBSCAN(eps=0.45, min_samples=5)
    model.fit(X)

    plot_all(X, model, save_path="output.png")
