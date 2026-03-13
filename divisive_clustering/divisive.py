import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# ------------------------------------------------------------------ #
#  Divisive (Top-Down) Hierarchical Clustering (DIANA-style)          #
# ------------------------------------------------------------------ #

class Divisive:
    def __init__(self, k: int = 3):
        """
        Divisive Hierarchical Clustering from scratch (DIANA-style).

        Starts with all points in one cluster, then recursively splits
        the cluster with the highest diameter until k clusters remain.

        Parameters:
            k : Number of final clusters
        """
        self.k = k
        self.labels = None
        self.split_history = []   # (cluster_size, diameter) at each split

    # ------------------------------------------------------------------ #
    #  Core helpers                                                        #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _diameter(points: np.ndarray) -> float:
        """Maximum pairwise distance within a set of points."""
        if len(points) <= 1:
            return 0.0
        diff = points[:, None] - points[None, :]
        dists = np.sqrt((diff ** 2).sum(axis=2))
        return float(dists.max())

    @staticmethod
    def _split_cluster(points: np.ndarray) -> tuple:
        """
        Split one cluster into two using a greedy splinter approach:
        1. Find the point most dissimilar to all others → start splinter group
        2. Iteratively move points from main group to splinter if they are
           closer to splinter mean than to main mean
        """
        if len(points) <= 1:
            return points, np.array([])

        # Step 1: find the most 'isolated' point
        avg_dists = np.mean(
            np.sqrt(((points[:, None] - points[None, :]) ** 2).sum(axis=2)),
            axis=1
        )
        splinter = [np.argmax(avg_dists)]
        main = list(set(range(len(points))) - set(splinter))

        # Step 2: greedily move points to splinter group
        changed = True
        while changed:
            changed = False
            splinter_mean = points[splinter].mean(axis=0)
            main_mean = points[main].mean(axis=0)
            to_move = []
            for idx in main:
                d_main = np.linalg.norm(points[idx] - main_mean)
                d_splinter = np.linalg.norm(points[idx] - splinter_mean)
                if d_splinter < d_main:
                    to_move.append(idx)
            if to_move:
                splinter.extend(to_move)
                main = list(set(main) - set(to_move))
                changed = True
            if len(main) == 0:
                break

        return points[main], points[splinter]

    # ------------------------------------------------------------------ #
    #  Fit                                                                 #
    # ------------------------------------------------------------------ #

    def fit(self, X: np.ndarray) -> "Divisive":
        X = np.array(X, dtype=float)
        n = len(X)

        # Track cluster membership by index
        clusters = [list(range(n))]   # start: one big cluster

        while len(clusters) < self.k:
            # Pick the cluster with the largest diameter to split
            diameters = [self._diameter(X[c]) for c in clusters]
            split_idx = int(np.argmax(diameters))
            chosen = clusters[split_idx]

            print(f"  Splitting cluster of size {len(chosen)}, diameter={diameters[split_idx]:.3f}")
            self.split_history.append((len(chosen), diameters[split_idx]))

            pts = X[chosen]
            main_pts, splinter_pts = self._split_cluster(pts)

            if len(splinter_pts) == 0:
                # Cannot split further; stop
                break

            # Recover original indices
            def find_indices(subset):
                idxs = []
                for pt in subset:
                    match = np.where((X == pt).all(axis=1))[0]
                    if len(match): idxs.append(match[0])
                return idxs

            main_idx = find_indices(main_pts)
            split_idx2 = find_indices(splinter_pts)

            clusters.pop(split_idx)
            clusters.append(main_idx)
            clusters.append(split_idx2)

        # Assign labels
        self.labels = np.zeros(n, dtype=int)
        for label, cluster in enumerate(clusters):
            for idx in cluster:
                self.labels[idx] = label

        return self

    def fit_predict(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).labels


# ------------------------------------------------------------------ #
#  Plotting                                                            #
# ------------------------------------------------------------------ #

PALETTE = ["#E63946", "#2A9D8F", "#457B9D", "#E9C46A",
           "#F4A261", "#6A4C93", "#43AA8B", "#F8961E"]


def plot_all(X: np.ndarray, model: Divisive, save_path: str = None):
    fig = plt.figure(figsize=(14, 5), facecolor="#0F1117")
    fig.suptitle(f"Divisive Hierarchical Clustering  (k={model.k})",
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

    # ── Split diameters ───────────────────────────────────────────────
    ax2 = fig.add_subplot(gs[1])
    ax2.set_facecolor("#1A1D27")
    steps = range(1, len(model.split_history) + 1)
    diams = [d for _, d in model.split_history]
    sizes = [s for s, _ in model.split_history]
    ax2.bar(steps, diams, color="#457B9D", alpha=0.8, label="Diameter at split")
    ax2b = ax2.twinx()
    ax2b.plot(steps, sizes, color="#E9C46A", linewidth=2, marker="o", label="Cluster size")
    ax2b.tick_params(colors="gray")
    ax2b.set_ylabel("Cluster Size", color="#E9C46A")
    ax2.set_title("Split History", color="white", fontsize=12)
    ax2.set_xlabel("Split Step", color="gray")
    ax2.set_ylabel("Diameter", color="gray")
    ax2.tick_params(colors="gray")
    for sp in ax2.spines.values(): sp.set_edgecolor("#333")
    lines1, labels1 = ax2.get_legend_handles_labels()
    lines2, labels2 = ax2b.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2, fontsize=8,
               labelcolor="white", facecolor="#1A1D27", edgecolor="#333")

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
        np.random.randn(70, 2) * 0.9 + center
        for center in [(0, 0), (6, 0), (3, 6)]
    ])

    print("Fitting Divisive Clustering...")
    model = Divisive(k=3)
    model.fit(X)

    plot_all(X, model, save_path="output.png")
