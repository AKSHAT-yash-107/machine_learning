import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# ------------------------------------------------------------------ #
#  K-Medoids (PAM — Partitioning Around Medoids)                      #
# ------------------------------------------------------------------ #

class KMedoids:
    def __init__(self, k: int = 3, max_iters: int = 100, init: str = "kmeans++"):
        """
        K-Medoids clustering (PAM algorithm).

        Unlike K-Means, cluster centers are actual data points (medoids),
        making it robust to outliers.

        Parameters:
            k         : Number of clusters
            max_iters : Maximum swap iterations
            init      : 'kmeans++' or 'random'
        """
        self.k = k
        self.max_iters = max_iters
        self.init = init
        self.medoid_indices = None
        self.medoids = None
        self.labels = None
        self.cost = None
        self._cost_history = []

    # ------------------------------------------------------------------ #
    #  Distance                                                            #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _dist_matrix(X: np.ndarray) -> np.ndarray:
        """Compute full pairwise Euclidean distance matrix."""
        diff = X[:, None, :] - X[None, :, :]
        return np.sqrt((diff ** 2).sum(axis=2))

    # ------------------------------------------------------------------ #
    #  Initialisation                                                      #
    # ------------------------------------------------------------------ #

    def _init_random(self, n: int) -> np.ndarray:
        return np.random.choice(n, self.k, replace=False)

    def _init_kmeanspp(self, X: np.ndarray, D: np.ndarray) -> np.ndarray:
        indices = [np.random.randint(len(X))]
        for _ in range(1, self.k):
            dists = D[:, indices].min(axis=1) ** 2
            probs = dists / dists.sum()
            indices.append(np.random.choice(len(X), p=probs))
        return np.array(indices)

    # ------------------------------------------------------------------ #
    #  Core PAM steps                                                      #
    # ------------------------------------------------------------------ #

    def _assign(self, D: np.ndarray, medoid_idx: np.ndarray) -> np.ndarray:
        return np.argmin(D[:, medoid_idx], axis=1)

    def _total_cost(self, D: np.ndarray, medoid_idx: np.ndarray) -> float:
        labels = self._assign(D, medoid_idx)
        return float(sum(D[i, medoid_idx[labels[i]]] for i in range(len(D))))

    # ------------------------------------------------------------------ #
    #  Public API                                                          #
    # ------------------------------------------------------------------ #

    def fit(self, X: np.ndarray) -> "KMedoids":
        X = np.array(X, dtype=float)
        n = len(X)
        D = self._dist_matrix(X)
        self._cost_history = []

        med_idx = (
            self._init_kmeanspp(X, D) if self.init == "kmeans++"
            else self._init_random(n)
        )

        best_cost = self._total_cost(D, med_idx)
        self._cost_history.append(best_cost)

        for iteration in range(self.max_iters):
            improved = False
            for m in range(self.k):
                for candidate in range(n):
                    if candidate in med_idx:
                        continue
                    new_idx = med_idx.copy()
                    new_idx[m] = candidate
                    new_cost = self._total_cost(D, new_idx)
                    if new_cost < best_cost - 1e-9:
                        best_cost = new_cost
                        med_idx = new_idx
                        improved = True

            self._cost_history.append(best_cost)
            if not improved:
                print(f"  Converged at iteration {iteration + 1}")
                break

        self.medoid_indices = med_idx
        self.medoids = X[med_idx]
        self.labels = self._assign(D, med_idx)
        self.cost = best_cost
        return self

    def predict(self, X_new: np.ndarray) -> np.ndarray:
        X_new = np.array(X_new, dtype=float)
        dists = np.linalg.norm(X_new[:, None] - self.medoids[None, :], axis=2)
        return np.argmin(dists, axis=1)

    def fit_predict(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).labels


# ------------------------------------------------------------------ #
#  Plotting                                                            #
# ------------------------------------------------------------------ #

PALETTE = ["#E63946", "#2A9D8F", "#457B9D", "#E9C46A",
           "#F4A261", "#6A4C93", "#43AA8B", "#F8961E"]


def plot_all(X: np.ndarray, km: KMedoids, save_path: str = None):
    fig = plt.figure(figsize=(14, 5), facecolor="#0F1117")
    fig.suptitle("K-Medoids (PAM) Clustering", fontsize=15, color="white")
    gs = gridspec.GridSpec(1, 2, figure=fig, wspace=0.35)

    # ── Cluster scatter ───────────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0])
    ax1.set_facecolor("#1A1D27")
    for j in range(km.k):
        pts = X[km.labels == j]
        ax1.scatter(pts[:, 0], pts[:, 1], color=PALETTE[j % len(PALETTE)],
                    alpha=0.65, s=45, edgecolors="none", label=f"Cluster {j}")
    ax1.scatter(km.medoids[:, 0], km.medoids[:, 1],
                c="white", s=250, marker="*", zorder=10, label="Medoids")
    ax1.set_title("Cluster Assignments", color="white", fontsize=12)
    ax1.tick_params(colors="gray")
    for sp in ax1.spines.values(): sp.set_edgecolor("#333")
    ax1.legend(fontsize=8, labelcolor="white", facecolor="#1A1D27", edgecolor="#333")

    # ── Cost convergence ──────────────────────────────────────────────
    ax2 = fig.add_subplot(gs[1])
    ax2.set_facecolor("#1A1D27")
    ax2.plot(range(len(km._cost_history)), km._cost_history,
             color="#E63946", linewidth=2, marker="o", markersize=5)
    ax2.set_title("Cost per Iteration", color="white", fontsize=12)
    ax2.set_xlabel("Iteration", color="gray")
    ax2.set_ylabel("Total Cost (Sum of Distances)", color="gray")
    ax2.tick_params(colors="gray")
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
        np.random.randn(80, 2) * 0.8 + center
        for center in [(0, 0), (6, 1), (3, 7)]
    ])
    # Add outliers to show robustness
    outliers = np.array([[15, 15], [-8, 10], [10, -5]])
    X = np.vstack([X, outliers])

    print("Fitting K-Medoids (PAM)...")
    km = KMedoids(k=3, init="kmeans++")
    km.fit(X)
    print(f"  Final cost  : {km.cost:.4f}")
    print(f"  Medoids idx : {km.medoid_indices}")
    print(f"  Medoids     :\n{km.medoids}")

    plot_all(X, km, save_path="output.png")
