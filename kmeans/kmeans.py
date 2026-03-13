import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


class KMeans:
    def __init__(self, k: int = 3, max_iters: int = 100, tol: float = 1e-4, init: str = "kmeans++"):
        """
        K-Means Clustering from scratch.

        Parameters:
            k         : Number of clusters
            max_iters : Maximum iterations before stopping
            tol       : Convergence tolerance (centroid shift)
            init      : 'kmeans++' or 'random'
        """
        self.k = k
        self.max_iters = max_iters
        self.tol = tol
        self.init = init
        self.centroids = None
        self.labels = None
        self.inertia = None
        self._inertia_history = []

    # ------------------------------------------------------------------ #
    #  Initialisation                                                      #
    # ------------------------------------------------------------------ #

    def _init_random(self, X: np.ndarray) -> np.ndarray:
        idx = np.random.choice(len(X), self.k, replace=False)
        return X[idx].copy()

    def _init_kmeans_plus_plus(self, X: np.ndarray) -> np.ndarray:
        """K-Means++ spreads initial centroids to speed up convergence."""
        centroids = [X[np.random.randint(len(X))]]
        for _ in range(1, self.k):
            dists = np.array([
                min(np.linalg.norm(x - c) ** 2 for c in centroids) for x in X
            ])
            probs = dists / dists.sum()
            centroids.append(X[np.random.choice(len(X), p=probs)])
        return np.array(centroids)

    # ------------------------------------------------------------------ #
    #  Core steps                                                          #
    # ------------------------------------------------------------------ #

    def _assign_labels(self, X: np.ndarray) -> np.ndarray:
        dists = np.linalg.norm(X[:, None] - self.centroids[None, :], axis=2)
        return np.argmin(dists, axis=1)

    def _update_centroids(self, X: np.ndarray) -> np.ndarray:
        return np.array([
            X[self.labels == j].mean(axis=0) if (self.labels == j).any() else self.centroids[j]
            for j in range(self.k)
        ])

    def _compute_inertia(self, X: np.ndarray) -> float:
        return float(sum(
            np.sum((X[self.labels == j] - self.centroids[j]) ** 2)
            for j in range(self.k)
        ))

    # ------------------------------------------------------------------ #
    #  Public API                                                          #
    # ------------------------------------------------------------------ #

    def fit(self, X: np.ndarray) -> "KMeans":
        X = np.array(X, dtype=float)
        self._inertia_history = []

        self.centroids = (
            self._init_kmeans_plus_plus(X) if self.init == "kmeans++"
            else self._init_random(X)
        )

        for iteration in range(self.max_iters):
            self.labels = self._assign_labels(X)
            new_centroids = self._update_centroids(X)
            self._inertia_history.append(self._compute_inertia(X))

            shift = np.linalg.norm(new_centroids - self.centroids)
            self.centroids = new_centroids
            if shift < self.tol:
                print(f"  Converged at iteration {iteration + 1}")
                break

        self.inertia = self._compute_inertia(X)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self._assign_labels(np.array(X, dtype=float))

    def fit_predict(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).labels

    # ------------------------------------------------------------------ #
    #  Elbow method                                                        #
    # ------------------------------------------------------------------ #

    @staticmethod
    def find_elbow(X: np.ndarray, k_range=range(1, 11)) -> list:
        inertias = []
        for k in k_range:
            km = KMeans(k=k)
            km.fit(X)
            inertias.append(km.inertia)
        return inertias


# ------------------------------------------------------------------ #
#  Plotting                                                            #
# ------------------------------------------------------------------ #

PALETTE = [
    "#E63946", "#2A9D8F", "#457B9D", "#E9C46A",
    "#F4A261", "#6A4C93", "#43AA8B", "#F8961E",
]


def plot_all(X: np.ndarray, km: KMeans, elbow_inertias: list, k_range: range, save_path: str = None):
    fig = plt.figure(figsize=(16, 5), facecolor="#0F1117")
    fig.suptitle("K-Means Clustering — Results", fontsize=16, color="white", y=1.01)

    gs = gridspec.GridSpec(1, 3, figure=fig, wspace=0.35)

    # ── Panel 1: Cluster scatter ──────────────────────────────────────
    ax1 = fig.add_subplot(gs[0])
    ax1.set_facecolor("#1A1D27")
    for j in range(km.k):
        pts = X[km.labels == j]
        ax1.scatter(pts[:, 0], pts[:, 1], color=PALETTE[j % len(PALETTE)],
                    alpha=0.7, s=40, edgecolors="none", label=f"Cluster {j}")
    ax1.scatter(km.centroids[:, 0], km.centroids[:, 1],
                c="white", s=220, marker="*", zorder=10, label="Centroids")
    ax1.set_title("Cluster Assignments", color="white", fontsize=12)
    ax1.tick_params(colors="gray")
    for sp in ax1.spines.values():
        sp.set_edgecolor("#333")
    ax1.legend(fontsize=8, labelcolor="white", facecolor="#1A1D27", edgecolor="#333")

    # ── Panel 2: Inertia convergence ─────────────────────────────────
    ax2 = fig.add_subplot(gs[1])
    ax2.set_facecolor("#1A1D27")
    ax2.plot(range(1, len(km._inertia_history) + 1), km._inertia_history,
             color="#2A9D8F", linewidth=2, marker="o", markersize=4)
    ax2.set_title("Inertia per Iteration", color="white", fontsize=12)
    ax2.set_xlabel("Iteration", color="gray")
    ax2.set_ylabel("WCSS (Inertia)", color="gray")
    ax2.tick_params(colors="gray")
    for sp in ax2.spines.values():
        sp.set_edgecolor("#333")

    # ── Panel 3: Elbow curve ──────────────────────────────────────────
    ax3 = fig.add_subplot(gs[2])
    ax3.set_facecolor("#1A1D27")
    ax3.plot(list(k_range), elbow_inertias, color="#E63946", linewidth=2,
             marker="o", markersize=5)
    ax3.axvline(x=km.k, color="white", linestyle="--", linewidth=1, alpha=0.6, label=f"k={km.k}")
    ax3.set_title("Elbow Method", color="white", fontsize=12)
    ax3.set_xlabel("Number of Clusters (k)", color="gray")
    ax3.set_ylabel("Inertia", color="gray")
    ax3.tick_params(colors="gray")
    ax3.legend(fontsize=9, labelcolor="white", facecolor="#1A1D27", edgecolor="#333")
    for sp in ax3.spines.values():
        sp.set_edgecolor("#333")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
        print(f"  Saved plot → {save_path}")
    plt.show()


# ------------------------------------------------------------------ #
#  Main demo                                                           #
# ------------------------------------------------------------------ #

if __name__ == "__main__":
    np.random.seed(42)

    # Generate synthetic blobs
    X = np.vstack([
        np.random.randn(120, 2) * 0.9 + center
        for center in [(0, 0), (6, 1), (3, 7), (8, 6)]
    ])

    K = 4
    k_range = range(1, 11)

    print(f"Fitting K-Means with k={K} ...")
    km = KMeans(k=K, init="kmeans++")
    km.fit(X)
    print(f"  Final inertia : {km.inertia:.2f}")
    print(f"  Centroids     :\n{km.centroids}")

    print("Running elbow method ...")
    elbow_inertias = KMeans.find_elbow(X, k_range)

    plot_all(X, km, elbow_inertias, k_range, save_path="output.png")
