import math

import numpy as np
from joblib import Parallel, delayed

__all__ = ["HarrisForest"]


class _Node:
    """Tree node: split (feature/threshold/children) or leaf (mean_p/mean_r)."""

    def __init__(self, is_leaf, n_samples,
                 feature=None, threshold=None, left=None, right=None,
                 mean_p=None, mean_r=None):
        self.is_leaf = is_leaf
        self.n_samples = n_samples
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.mean_p = mean_p
        self.mean_r = mean_r


def _resolve_max_features(max_features, n_features):
    """Turn the user-facing ``max_features`` spec into an integer count."""
    if max_features is None:
        return n_features
    if isinstance(max_features, str):
        if max_features == "sqrt":
            return max(1, int(round(math.sqrt(n_features))))
        if max_features == "log2":
            return max(1, int(round(math.log2(n_features))))
        raise ValueError(f"unknown max_features string: {max_features!r}")
    if isinstance(max_features, float):
        return max(1, min(n_features, int(round(max_features * n_features))))
    if isinstance(max_features, (int, np.integer)):
        return max(1, min(n_features, int(max_features)))
    raise TypeError(f"unsupported max_features type: {type(max_features)}")


def _node_sum_of_squares(Y):
    """Total sum of squares of Y around its column means."""
    n = Y.shape[0]
    if n == 0:
        return 0.0
    col_sum = Y.sum(axis=0)
    return float(np.sum(Y * Y) - np.sum(col_sum * col_sum) / n)


def _best_split_for_feature(
    x_col,
    P_std_node,
    R_std_node,
    lam,
    min_samples_leaf,
):
    """Sweep all thresholds of one feature via cumulative sums. Returns (score, threshold, order) or None."""
    n = x_col.shape[0]
    order = np.argsort(x_col, kind="mergesort")  # stable -> deterministic
    sv = x_col[order]

    distinct = sv[:-1] != sv[1:]
    if not distinct.any():
        return None

    left_n = np.arange(1, n)          # candidate |S_L|, one per boundary i
    right_n = n - left_n
    valid = distinct & (left_n >= min_samples_leaf) & (right_n >= min_samples_leaf)
    if not valid.any():
        return None

    Ps = P_std_node[order]
    Rs = R_std_node[order]

    cs_p = np.cumsum(Ps, axis=0)
    csq_p = np.cumsum(Ps * Ps, axis=0)
    cs_r = np.cumsum(Rs, axis=0)
    csq_r = np.cumsum(Rs * Rs, axis=0)

    total_p, total_sq_p = cs_p[-1], csq_p[-1]
    total_r, total_sq_r = cs_r[-1], csq_r[-1]

    left_sum_p, left_sumsq_p = cs_p[:-1], csq_p[:-1]
    left_sum_r, left_sumsq_r = cs_r[:-1], csq_r[:-1]
    right_sum_p = total_p - left_sum_p
    right_sumsq_p = total_sq_p - left_sumsq_p
    right_sum_r = total_r - left_sum_r
    right_sumsq_r = total_sq_r - left_sumsq_r

    ln = left_n[:, None].astype(np.float64)
    rn = right_n[:, None].astype(np.float64)

    left_ss_p = left_sumsq_p.sum(axis=1) - (left_sum_p ** 2 / ln).sum(axis=1)
    right_ss_p = right_sumsq_p.sum(axis=1) - (right_sum_p ** 2 / rn).sum(axis=1)
    left_ss_r = left_sumsq_r.sum(axis=1) - (left_sum_r ** 2 / ln).sum(axis=1)
    right_ss_r = right_sumsq_r.sum(axis=1) - (right_sum_r ** 2 / rn).sum(axis=1)

    combined = lam * (left_ss_r + right_ss_r) + (1.0 - lam) * (left_ss_p + right_ss_p)
    combined = np.where(valid, combined, np.inf)

    i = int(np.argmin(combined))
    if not np.isfinite(combined[i]):
        return None

    threshold = (sv[i] + sv[i + 1]) / 2.0
    return float(combined[i]), threshold, order


class _HarrisTree:
    """A single tree in the HARRIS forest, grown greedily top-down."""

    def __init__(self, lam, max_features, min_samples_leaf, max_depth):
        self.lam = lam
        self.max_features = max_features
        self.min_samples_leaf = min_samples_leaf
        self.max_depth = max_depth if max_depth is not None else np.iinfo(np.int64).max
        self.root = None

    def fit(self, X, P, R, P_std, R_std, sample_idx, rng):
        """Grow the tree on the rows in sample_idx; P_std/R_std drive the splits, P/R the leaf means."""
        n_features = X.shape[1]
        mtry = _resolve_max_features(self.max_features, n_features)
        self.root = self._build(X, P, R, P_std, R_std, sample_idx, depth=0, mtry=mtry, rng=rng)
        return self

    def _build(self, X, P, R, P_std, R_std, idx, depth, mtry, rng):
        n = idx.shape[0]

        def make_leaf():
            return _Node(
                is_leaf=True,
                n_samples=n,
                mean_p=P[idx].mean(axis=0),
                mean_r=R[idx].mean(axis=0),
            )

        if n < 2 * self.min_samples_leaf or depth >= self.max_depth:
            return make_leaf()

        Pn_std, Rn_std = P_std[idx], R_std[idx]
        parent_ss = self.lam * _node_sum_of_squares(Rn_std) + \
            (1.0 - self.lam) * _node_sum_of_squares(Pn_std)

        if parent_ss <= 1e-12:
            return make_leaf()  # node already pure on the combined objective

        n_features = X.shape[1]
        feat_candidates = rng.choice(n_features, size=min(mtry, n_features), replace=False)

        best = None  # (score, feature, threshold, order)
        for f in feat_candidates:
            result = _best_split_for_feature(
                X[idx, f], Pn_std, Rn_std, self.lam, self.min_samples_leaf
            )
            if result is None:
                continue
            score, threshold, order = result
            if best is None or score < best[0]:
                best = (score, f, threshold, order)

        if best is None:
            return make_leaf()

        score, feature, threshold, order = best
        eps = 1e-9 * max(1.0, abs(parent_ss))
        if score >= parent_ss - eps:
            return make_leaf()

        sorted_idx = idx[order]

        x_sorted = X[sorted_idx, feature]
        left_mask = x_sorted <= threshold
        left_idx = sorted_idx[left_mask]
        right_idx = sorted_idx[~left_mask]

        left_node = self._build(X, P, R, P_std, R_std, left_idx, depth + 1, mtry, rng)
        right_node = self._build(X, P, R, P_std, R_std, right_idx, depth + 1, mtry, rng)

        return _Node(
            is_leaf=False,
            n_samples=n,
            feature=int(feature),
            threshold=float(threshold),
            left=left_node,
            right=right_node,
        )

    def predict(self, X, n_algorithms):
        """Return (mean_p, mean_r), each (n_query, n_algorithms)."""
        n = X.shape[0]
        out_p = np.empty((n, n_algorithms), dtype=np.float64)
        out_r = np.empty((n, n_algorithms), dtype=np.float64)
        self._predict_into(self.root, X, np.arange(n), out_p, out_r)
        return out_p, out_r

    def _predict_into(self, node, X, idx, out_p, out_r):
        if node.is_leaf:
            out_p[idx] = node.mean_p
            out_r[idx] = node.mean_r
            return
        vals = X[idx, node.feature]
        mask = vals <= node.threshold
        if mask.any():
            self._predict_into(node.left, X, idx[mask], out_p, out_r)
        if (~mask).any():
            self._predict_into(node.right, X, idx[~mask], out_p, out_r)


def _fit_one_tree(t, lam, max_features, min_samples_leaf, max_depth,
                   bootstrap, n, X, P, R, P_std, R_std, seed):
    """Module-level worker so joblib can pickle it for n_jobs > 1."""
    rng = np.random.default_rng(seed)
    if bootstrap:
        sample_idx = rng.integers(0, n, size=n)
    else:
        sample_idx = np.arange(n)
    tree = _HarrisTree(lam, max_features, min_samples_leaf, max_depth)
    tree.fit(X, P, R, P_std, R_std, sample_idx, rng)
    return tree


class HarrisForest:
    """Hybrid ranking/regression random forest: split loss mixes ranking (lam) and regression (1-lam)."""

    def __init__(self, lam=0.5, n_estimators=100, max_features="sqrt",
                 min_samples_leaf=3, max_depth=None, bootstrap=True,
                 random_state=None, n_jobs=1):
        if not (0.0 <= lam <= 1.0):
            raise ValueError("lam must be in [0, 1]")
        self.lam = lam
        self.n_estimators = n_estimators
        self.max_features = max_features
        self.min_samples_leaf = min_samples_leaf
        self.max_depth = max_depth
        self.bootstrap = bootstrap
        self.random_state = random_state
        self.n_jobs = n_jobs

    def fit(self, X, P, R):
        """Fit on X (n, d) and P, R (n, m); accepts arrays or DataFrames."""
        X = np.asarray(X, dtype=np.float64)
        P = np.asarray(P, dtype=np.float64)
        R = np.asarray(R, dtype=np.float64)
        if X.ndim != 2 or P.ndim != 2 or R.ndim != 2:
            raise ValueError("X, P, R must all be 2D")
        n = X.shape[0]
        if P.shape[0] != n or R.shape[0] != n:
            raise ValueError("X, P, R must have the same number of rows")
        if P.shape != R.shape:
            raise ValueError("P and R must have the same shape (n, n_algorithms)")

        self.n_features_in_ = X.shape[1]
        self.n_algorithms_ = P.shape[1]

        # z-score P and R once, globally, so their scales are comparable before mixing by lam
        self.P_mean_ = P.mean(axis=0)
        self.P_std_ = P.std(axis=0)
        self.P_std_ = np.where(self.P_std_ == 0, 1.0, self.P_std_)
        self.R_mean_ = R.mean(axis=0)
        self.R_std_ = R.std(axis=0)
        self.R_std_ = np.where(self.R_std_ == 0, 1.0, self.R_std_)

        P_std = (P - self.P_mean_) / self.P_std_
        R_std = (R - self.R_mean_) / self.R_std_

        seed_seq = np.random.SeedSequence(self.random_state)
        child_seeds = seed_seq.spawn(self.n_estimators)

        jobs = (
            delayed(_fit_one_tree)(
                t, self.lam, self.max_features, self.min_samples_leaf,
                self.max_depth, self.bootstrap, n, X, P, R, P_std, R_std,
                child_seeds[t],
            )
            for t in range(self.n_estimators)
        )
        self.trees_ = Parallel(n_jobs=self.n_jobs)(jobs)
        return self

    def _predict_all(self, X):
        X = np.asarray(X, dtype=np.float64)
        n_q = X.shape[0]
        sum_p = np.zeros((n_q, self.n_algorithms_), dtype=np.float64)
        sum_r = np.zeros((n_q, self.n_algorithms_), dtype=np.float64)
        for tree in self.trees_:
            p, r = tree.predict(X, self.n_algorithms_)
            sum_p += p
            sum_r += r
        n_trees = len(self.trees_)
        return sum_p / n_trees, sum_r / n_trees

    def predict_perf(self, X):
        """Predicted performance (n, m), averaged over trees."""
        perf, _ = self._predict_all(X)
        return perf

    def predict_rank(self, X):
        """Predicted rank scores (n, m), averaged over trees; lower = better."""
        _, rank = self._predict_all(X)
        return rank


def _smoke_test():
    import os
    import time

    import pandas as pd
    from scipy.stats import rankdata, spearmanr

    repo_root = os.path.dirname(os.path.abspath(__file__))
    X_df = pd.read_csv(os.path.join(repo_root, "data", "X.csv")).set_index("dataset_id")
    P_df = pd.read_csv(os.path.join(repo_root, "data", "P.csv")).set_index("dataset_id")
    X_df, P_df = X_df.align(P_df, join="inner", axis=0)

    X = X_df.to_numpy(dtype=np.float64)
    P = P_df.to_numpy(dtype=np.float64)
    # rank 1 = best performance -> rankdata ascending on -P
    R = rankdata(-P, method="average", axis=1)

    n, d = X.shape
    m = P.shape[1]
    print(f"Loaded data: X {X.shape}, P {P.shape}, R {R.shape}")

    rng = np.random.default_rng(0)

    # 1. lam=0 must ignore R (predict_perf unaffected by replacing R with noise); lam=1 symmetric for P
    R_noise = rng.normal(size=R.shape)
    P_noise = rng.normal(size=P.shape)

    f_lam0_a = HarrisForest(lam=0.0, n_estimators=20, random_state=42, n_jobs=1).fit(X, P, R)
    f_lam0_b = HarrisForest(lam=0.0, n_estimators=20, random_state=42, n_jobs=1).fit(X, P, R_noise)
    same_perf_lam0 = np.allclose(f_lam0_a.predict_perf(X), f_lam0_b.predict_perf(X))
    assert same_perf_lam0, "lam=0 predict_perf changed when R was replaced by noise!"
    print(f"[check 1a] lam=0 predict_perf invariant to R content: {same_perf_lam0}")

    f_lam1_a = HarrisForest(lam=1.0, n_estimators=20, random_state=42, n_jobs=1).fit(X, P, R)
    f_lam1_b = HarrisForest(lam=1.0, n_estimators=20, random_state=42, n_jobs=1).fit(X, P_noise, R)
    same_rank_lam1 = np.allclose(f_lam1_a.predict_rank(X), f_lam1_b.predict_rank(X))
    assert same_rank_lam1, "lam=1 predict_rank changed when P was replaced by noise!"
    print(f"[check 1b] lam=1 predict_rank invariant to P content: {same_rank_lam1}")

    # 2. lam sweep, 45 train / 5 held-out, mean Spearman(predict_rank, R)
    perm = rng.permutation(n)
    train_idx, test_idx = perm[:45], perm[45:]
    print(f"\nlam sweep, train={len(train_idx)} test={len(test_idx)}:")
    for lam in (0.0, 0.25, 0.5, 0.75, 1.0):
        forest = HarrisForest(lam=lam, n_estimators=100, random_state=0, n_jobs=6)
        forest.fit(X[train_idx], P[train_idx], R[train_idx])
        pred_rank = forest.predict_rank(X[test_idx])
        corrs = [spearmanr(pred_rank[i], R[test_idx[i]]).statistic for i in range(len(test_idx))]
        mean_corr = float(np.nanmean(corrs))
        print(f"  lam={lam:>4}: mean Spearman(predict_rank, true rank) = {mean_corr:.4f}  (per-dataset: {[round(c,3) for c in corrs]})")

    # 3. reproducibility: same random_state -> bit-identical predict_rank
    fa = HarrisForest(lam=0.5, n_estimators=50, random_state=7, n_jobs=1).fit(X, P, R)
    fb = HarrisForest(lam=0.5, n_estimators=50, random_state=7, n_jobs=1).fit(X, P, R)
    identical = np.array_equal(fa.predict_rank(X), fb.predict_rank(X))
    assert identical, "same random_state did not give bit-identical predict_rank!"
    print(f"\n[check 3] same random_state -> bit-identical predict_rank: {identical}")

    # n_jobs shouldn't change reproducibility either
    fc = HarrisForest(lam=0.5, n_estimators=50, random_state=7, n_jobs=6).fit(X, P, R)
    identical_njobs = np.array_equal(fa.predict_rank(X), fc.predict_rank(X))
    print(f"[check 3b] n_jobs=1 vs n_jobs=6, same random_state -> bit-identical: {identical_njobs}")

    # 4. timing, ~49 datasets, 100 trees
    bench_idx = perm[:49]
    t0 = time.time()
    bench_forest = HarrisForest(lam=0.5, n_estimators=100, random_state=1, n_jobs=6)
    bench_forest.fit(X[bench_idx], P[bench_idx], R[bench_idx])
    elapsed = time.time() - t0
    print(f"\n[timing] fit 100 trees on {len(bench_idx)} datasets: {elapsed:.3f}s "
          f"(n_jobs=6).")


if __name__ == "__main__":
    _smoke_test()
