"""
Exercise C - Unsupervised Learning: clustering of the UCI Wine dataset.

Models compared:
    * K-Means (Lloyd's algorithm, k-means++ initialisation)
    * Gaussian Mixture Model (Expectation-Maximisation)
    * Agglomerative clustering (Ward linkage)

The class labels are never used to fit or select a model. They are only used
at the very end (external metrics ARI / NMI) to check whether the clusters
found without supervision match the three real cultivars.

Run:  python exercise_c.py
Output: figures in figures/C/ and metric tables printed to stdout
        (also written to figures/C/results.txt).
"""

import os

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.datasets import load_wine
from sklearn.decomposition import PCA
from sklearn.metrics import (
    adjusted_rand_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    normalized_mutual_info_score,
    silhouette_score,
)
from sklearn.metrics.cluster import contingency_matrix
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import StandardScaler

SEED = 42
FIG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures", "C")
K_RANGE = range(2, 9)
N_REPEATS = 30  # random splits used for the stability study

os.makedirs(FIG_DIR, exist_ok=True)
_report_lines = []


def report(line=""):
    print(line)
    _report_lines.append(line)


def savefig(name):
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, name), dpi=150, bbox_inches="tight")
    plt.close()


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------
class WardClustering:
    """Agglomerative clustering (Ward) with a predict() for unseen points.

    Agglomerative clustering only labels the points it was fitted on. New
    points are assigned to the cluster with the nearest centroid, which is
    consistent with Ward's criterion (minimum increase of within-cluster
    variance, i.e. squared distance to the centroid).
    """

    def __init__(self, n_clusters, linkage="ward"):
        self.n_clusters = n_clusters
        self.linkage = linkage

    def fit(self, X):
        self.model_ = AgglomerativeClustering(n_clusters=self.n_clusters, linkage=self.linkage)
        self.labels_ = self.model_.fit_predict(X)
        self.centroids_ = np.array([X[self.labels_ == c].mean(axis=0) for c in range(self.n_clusters)])
        return self

    def predict(self, X):
        d = ((X[:, None, :] - self.centroids_[None, :, :]) ** 2).sum(axis=2)
        return d.argmin(axis=1)


def make_model(name, k, seed=SEED, covariance_type="full", linkage_type="ward"):
    if name == "K-Means":
        return KMeans(n_clusters=k, init="k-means++", n_init=10, random_state=seed)
    if name == "GMM":
        return GaussianMixture(n_components=k, covariance_type=covariance_type, n_init=5, random_state=seed)
    if name == "Agglomerative":
        return WardClustering(n_clusters=k, linkage=linkage_type)
    raise ValueError(name)


MODELS = ["K-Means", "GMM", "Agglomerative"]


def fit_predict(model, X):
    model.fit(X)
    return model.predict(X)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def internal_metrics(X, labels):
    if len(np.unique(labels)) < 2:
        return dict(silhouette=np.nan, davies_bouldin=np.nan, calinski_harabasz=np.nan)
    return dict(
        silhouette=silhouette_score(X, labels),
        davies_bouldin=davies_bouldin_score(X, labels),
        calinski_harabasz=calinski_harabasz_score(X, labels),
    )


def external_metrics(y_true, labels):
    return dict(
        ari=adjusted_rand_score(y_true, labels),
        nmi=normalized_mutual_info_score(y_true, labels),
        purity=matched_accuracy(y_true, labels),
    )


def matched_accuracy(y_true, labels):
    """Accuracy after the best one-to-one matching of clusters to classes
    (Hungarian algorithm on the contingency matrix)."""
    cm = contingency_matrix(y_true, labels)
    rows, cols = linear_sum_assignment(-cm)
    return cm[rows, cols].sum() / cm.sum()


def align_labels(y_true, labels):
    """Relabel clusters so that cluster c is matched to class c (for plots)."""
    cm = contingency_matrix(y_true, labels)
    rows, cols = linear_sum_assignment(-cm)
    classes, clusters = np.unique(y_true), np.unique(labels)
    mapping = {clusters[c]: classes[r] for r, c in zip(rows, cols)}
    unmatched = iter(range(len(classes), len(classes) + len(clusters)))
    return np.array([mapping[l] if l in mapping else next(unmatched) for l in labels])


# ---------------------------------------------------------------------------
# 1. Data
# ---------------------------------------------------------------------------
wine = load_wine()
X_raw, y = wine.data, wine.target
feature_names = [n.replace("od280/od315_of_diluted_wines", "OD280/OD315") for n in wine.feature_names]
class_names = ["cultivar 0", "cultivar 1", "cultivar 2"]

report("=" * 70)
report("DATA")
report("=" * 70)
report(f"Samples: {X_raw.shape[0]}, features: {X_raw.shape[1]}")
report(f"Class sizes: {np.bincount(y).tolist()}")
report()
report(f"{'feature':<22}{'mean':>10}{'std':>10}{'min':>10}{'max':>10}")
for i, name in enumerate(feature_names):
    col = X_raw[:, i]
    report(f"{name:<22}{col.mean():>10.2f}{col.std():>10.2f}{col.min():>10.2f}{col.max():>10.2f}")

# Train / test split: 80 / 20. The validation data are the held-out folds of a
# 5-fold cross-validation inside the training part (section 2): with only 142
# training points a single validation split of ~30 points is too noisy to
# choose the number of clusters.
# Stratification only keeps the cultivar proportions equal in both subsets so
# that the final external evaluation is representative. The labels are not
# used for fitting or for choosing hyper-parameters.
X_train_raw, X_test_raw, y_train, y_test = train_test_split(
    X_raw, y, test_size=0.2, stratify=y, random_state=SEED
)
report()
report(f"Split sizes - train: {len(X_train_raw)} (5-fold CV for validation), test: {len(X_test_raw)}")

# Z-score standardisation, statistics estimated on the training set only.
scaler = StandardScaler().fit(X_train_raw)
X_train = scaler.transform(X_train_raw)
X_test = scaler.transform(X_test_raw)

# PCA fitted on the training data, used only for 2D visualisation.
pca = PCA(n_components=2, random_state=SEED).fit(X_train)
report(f"PCA explained variance (PC1, PC2): {np.round(pca.explained_variance_ratio_, 3).tolist()}"
       f" -> {pca.explained_variance_ratio_.sum():.3f} total")

# --- Figure: PCA projection of the split, coloured by true class ---
fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
for subset, Xs, marker in [("train", X_train, "o"), ("test", X_test, "^")]:
    Z = pca.transform(Xs)
    axes[0].scatter(Z[:, 0], Z[:, 1], marker=marker, s=25, alpha=0.8, label=subset)
for c in range(3):
    Z = pca.transform(scaler.transform(X_raw[y == c]))
    axes[1].scatter(Z[:, 0], Z[:, 1], s=25, alpha=0.8, label=class_names[c])
for ax, title in zip(axes, ["Train / test split", "True cultivars (not used for training)"]):
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title(title)
    ax.legend()
savefig("C_pca_data.png")


# ---------------------------------------------------------------------------
# 2. Model selection with 5-fold cross-validation (no labels used)
# ---------------------------------------------------------------------------
# For every fold: standardise with the fold-train statistics, fit on the
# fold-train part, assign the held-out fold and score it with internal metrics.
report()
report("=" * 70)
report("MODEL SELECTION (5-fold CV on the training set, internal metrics only)")
report("=" * 70)

folds = list(KFold(n_splits=5, shuffle=True, random_state=SEED).split(X_train_raw))
COV_TYPES = ["full", "tied", "diag", "spherical"]


def cv_scores(build):
    """Average validation metrics of model build() over the CV folds."""
    scores = {"silhouette": [], "davies_bouldin": [], "calinski_harabasz": [], "loglik": [], "bic": []}
    for tr, va in folds:
        sc = StandardScaler().fit(X_train_raw[tr])
        Xtr, Xva = sc.transform(X_train_raw[tr]), sc.transform(X_train_raw[va])
        model = build().fit(Xtr)
        for metric, value in internal_metrics(Xva, model.predict(Xva)).items():
            scores[metric].append(value)
        if isinstance(model, GaussianMixture):
            scores["loglik"].append(model.score(Xva))
            scores["bic"].append(model.bic(Xtr))
    return {m: np.nanmean(v) if v else np.nan for m, v in scores.items()}


selection = {m: {"silhouette": [], "davies_bouldin": [], "calinski_harabasz": []} for m in MODELS}
for k in K_RANGE:
    for name in MODELS:
        s = cv_scores(lambda: make_model(name, k))
        for metric in selection[name]:
            selection[name][metric].append(s[metric])

# K-Means elbow on the whole training set
inertia = [make_model("K-Means", k).fit(X_train).inertia_ for k in K_RANGE]

report("Mean over the 5 validation folds (sil: higher is better, DB: lower, CH: higher)")
report(f"{'k':>3} | " + " | ".join(f"{m:^26}" for m in MODELS))
report(f"{'':>3} | " + " | ".join(f"{'sil':>8}{'DB':>9}{'CH':>9}" for _ in MODELS))
for i, k in enumerate(K_RANGE):
    row = " | ".join(
        f"{selection[m]['silhouette'][i]:>8.3f}{selection[m]['davies_bouldin'][i]:>9.3f}"
        f"{selection[m]['calinski_harabasz'][i]:>9.1f}"
        for m in MODELS
    )
    report(f"{k:>3} | {row}")

best_k = {m: list(K_RANGE)[int(np.nanargmax(selection[m]["silhouette"]))] for m in MODELS}
report()
report("k with the best CV silhouette: " + ", ".join(f"{m}: {k}" for m, k in best_k.items()))
report()
report(f"{'k':>3}{'K-Means inertia (train)':>26}")
for i, k in enumerate(K_RANGE):
    report(f"{k:>3}{inertia[i]:>26.1f}")

# GMM: number of components and covariance structure chosen jointly by the
# held-out log-likelihood (a proper validation criterion for a density model);
# BIC on the fold-train data is reported for comparison.
report()
report("GMM grid: mean held-out log-likelihood per sample (higher is better) / mean BIC (lower is better)")
report(f"{'k':>3} | " + " | ".join(f"{c:^18}" for c in COV_TYPES))
gmm_ll = np.zeros((len(K_RANGE), len(COV_TYPES)))
gmm_bic = np.zeros_like(gmm_ll)
for i, k in enumerate(K_RANGE):
    for j, cov in enumerate(COV_TYPES):
        s = cv_scores(lambda: make_model("GMM", k, covariance_type=cov))
        gmm_ll[i, j], gmm_bic[i, j] = s["loglik"], s["bic"]
    report(f"{k:>3} | " + " | ".join(f"{gmm_ll[i, j]:>8.2f} /{gmm_bic[i, j]:>7.0f}  " for j in range(len(COV_TYPES))))
i_ll, j_ll = np.unravel_index(np.argmax(gmm_ll), gmm_ll.shape)
i_bic, j_bic = np.unravel_index(np.argmin(gmm_bic), gmm_bic.shape)
report(f"  -> best held-out log-likelihood: k = {list(K_RANGE)[i_ll]}, covariance = {COV_TYPES[j_ll]}")
report(f"  -> best BIC:                     k = {list(K_RANGE)[i_bic]}, covariance = {COV_TYPES[j_bic]}")
best_cov = COV_TYPES[j_ll]

# Linkage choice for agglomerative clustering
report()
report("Agglomerative linkage selection (k = 3), CV silhouette and cluster sizes on the full training set:")
for link in ["ward", "complete", "average", "single"]:
    s = cv_scores(lambda: make_model("Agglomerative", 3, linkage_type=link))
    sizes = make_model("Agglomerative", 3, linkage_type=link).fit(X_train).labels_
    report(f"  {link:<10} silhouette = {s['silhouette']:.3f}   cluster sizes = {np.bincount(sizes).tolist()}")

# --- Figure: model selection curves ---
ks = list(K_RANGE)
fig, axes = plt.subplots(1, 4, figsize=(18, 4))
axes[0].plot(ks, inertia, "o-")
axes[0].set_title("K-Means: elbow (train inertia)")
axes[0].set_ylabel("within-cluster sum of squares")
for m in MODELS:
    axes[1].plot(ks, selection[m]["silhouette"], "o-", label=m)
    axes[2].plot(ks, selection[m]["davies_bouldin"], "o-", label=m)
axes[1].set_title("CV silhouette (higher = better)")
axes[2].set_title("CV Davies-Bouldin (lower = better)")
axes[1].legend()
axes[2].legend()
for j, cov in enumerate(COV_TYPES):
    axes[3].plot(ks, gmm_ll[:, j], "o-", label=cov)
axes[3].set_title("GMM: CV held-out log-likelihood")
axes[3].legend(title="covariance")
for ax in axes:
    ax.set_xlabel("number of clusters k")
    ax.axvline(3, color="grey", ls=":", lw=1)
savefig("C_model_selection.png")

# ---------------------------------------------------------------------------
# 3. Final models (k = 3) and evaluation on the test set
# ---------------------------------------------------------------------------
K_FINAL = 3
final = {
    "K-Means": make_model("K-Means", K_FINAL),
    "GMM": make_model("GMM", K_FINAL, covariance_type=best_cov),
    "Agglomerative": make_model("Agglomerative", K_FINAL),
}

report()
report("=" * 70)
report(f"FINAL MODELS (k = {K_FINAL}), fitted on train, evaluated on test")
report("=" * 70)
report(f"{'model':<15}{'sil':>8}{'DB':>8}{'CH':>8}{'ARI':>8}{'NMI':>8}{'acc*':>8}")
test_labels = {}
for name, model in final.items():
    model.fit(X_train)
    labels = model.predict(X_test)
    test_labels[name] = labels
    im_ = internal_metrics(X_test, labels)
    ex_ = external_metrics(y_test, labels)
    report(f"{name:<15}{im_['silhouette']:>8.3f}{im_['davies_bouldin']:>8.3f}{im_['calinski_harabasz']:>8.1f}"
           f"{ex_['ari']:>8.3f}{ex_['nmi']:>8.3f}{ex_['purity']:>8.3f}")
report("acc* = accuracy after optimal cluster-to-class matching (Hungarian algorithm)")

report()
report("Contingency matrices on the test set (rows = true cultivar, columns = cluster):")
for name, labels in test_labels.items():
    cm = contingency_matrix(y_test, align_labels(y_test, labels))
    report(f"  {name}:")
    for r in cm:
        report("    " + " ".join(f"{v:>4d}" for v in r))

# Agreement between the models themselves
report()
report("Agreement between models on the test set (ARI):")
for i, a in enumerate(MODELS):
    for b in MODELS[i + 1:]:
        report(f"  {a} vs {b}: {adjusted_rand_score(test_labels[a], test_labels[b]):.3f}")

# --- Figure: clusters in PCA space (train + test) ---
fig, axes = plt.subplots(1, 4, figsize=(19, 4.5))
Z_train, Z_test = pca.transform(X_train), pca.transform(X_test)
axes[0].scatter(Z_train[:, 0], Z_train[:, 1], c=y_train, cmap="viridis", s=20, alpha=0.5)
axes[0].scatter(Z_test[:, 0], Z_test[:, 1], c=y_test, cmap="viridis", s=45, marker="^", edgecolor="k")
axes[0].set_title("True cultivars")
for ax, name in zip(axes[1:], MODELS):
    model = final[name]
    tr = align_labels(y_train, model.predict(X_train))
    te = align_labels(y_test, test_labels[name])
    ax.scatter(Z_train[:, 0], Z_train[:, 1], c=tr, cmap="viridis", s=20, alpha=0.5)
    ax.scatter(Z_test[:, 0], Z_test[:, 1], c=te, cmap="viridis", s=45, marker="^", edgecolor="k")
    wrong = te != y_test
    ax.scatter(Z_test[wrong, 0], Z_test[wrong, 1], s=160, facecolors="none", edgecolors="red", lw=1.5)
    if name == "K-Means":
        C = pca.transform(model.cluster_centers_)
        ax.scatter(C[:, 0], C[:, 1], marker="X", s=200, c="red", edgecolor="k")
    if name == "GMM":
        C = pca.transform(model.means_)
        ax.scatter(C[:, 0], C[:, 1], marker="X", s=200, c="red", edgecolor="k")
    ax.set_title(f"{name} (test ARI = {adjusted_rand_score(y_test, test_labels[name]):.2f})")
for ax in axes:
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
fig.suptitle("Circles = train, triangles = test, red rings = test points in the wrong cluster", y=1.02)
savefig("C_clusters_pca.png")

# ---------------------------------------------------------------------------
# 4. Ablation: what happens without standardisation?
# ---------------------------------------------------------------------------
report()
report("=" * 70)
report("ABLATION: raw (unscaled) features vs z-score standardised features")
report("=" * 70)
report(f"{'model':<15}{'ARI raw':>10}{'ARI scaled':>12}{'NMI raw':>10}{'NMI scaled':>12}")
for name in MODELS:
    raw_labels = make_model(name, K_FINAL, covariance_type=best_cov).fit(X_train_raw).predict(X_test_raw)
    report(f"{name:<15}{adjusted_rand_score(y_test, raw_labels):>10.3f}"
           f"{adjusted_rand_score(y_test, test_labels[name]):>12.3f}"
           f"{normalized_mutual_info_score(y_test, raw_labels):>10.3f}"
           f"{normalized_mutual_info_score(y_test, test_labels[name]):>12.3f}")


# ---------------------------------------------------------------------------
# 5. Stability over random splits and initialisations
# ---------------------------------------------------------------------------
report()
report("=" * 70)
report(f"STABILITY: {N_REPEATS} random 80/20 splits (different seeds), k = 3")
report("=" * 70)
stab = {m: {"ari": [], "nmi": [], "sil": []} for m in MODELS}
for r in range(N_REPEATS):
    Xtr, Xte, ytr, yte = train_test_split(X_raw, y, test_size=0.2, stratify=y, random_state=r)
    sc_ = StandardScaler().fit(Xtr)
    Xtr, Xte = sc_.transform(Xtr), sc_.transform(Xte)
    for name in MODELS:
        labels = make_model(name, K_FINAL, seed=r, covariance_type=best_cov).fit(Xtr).predict(Xte)
        stab[name]["ari"].append(adjusted_rand_score(yte, labels))
        stab[name]["nmi"].append(normalized_mutual_info_score(yte, labels))
        stab[name]["sil"].append(silhouette_score(Xte, labels))
report(f"{'model':<15}{'ARI mean':>10}{'ARI std':>9}{'NMI mean':>10}{'NMI std':>9}{'sil mean':>10}{'sil std':>9}")
for name in MODELS:
    a, n, s = (np.array(stab[name][k]) for k in ("ari", "nmi", "sil"))
    report(f"{name:<15}{a.mean():>10.3f}{a.std():>9.3f}{n.mean():>10.3f}{n.std():>9.3f}{s.mean():>10.3f}{s.std():>9.3f}")

with open(os.path.join(FIG_DIR, "results.txt"), "w") as f:
    f.write("\n".join(_report_lines) + "\n")
