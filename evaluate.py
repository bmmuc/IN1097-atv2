import numpy as np
import pandas as pd
from scipy.stats import friedmanchisquare, rankdata, spearmanr, studentized_range

# Nemenyi q_alpha (alpha=0.05), Demsar (2006), JMLR 7:1-30, Table 5(b).
NEMENYI_Q_ALPHA_005 = {
    2: 1.960, 3: 2.343, 4: 2.569, 5: 2.728, 6: 2.850,
    7: 2.949, 8: 3.031, 9: 3.102, 10: 3.164,
}


def nemenyi_q_alpha(k, alpha=0.05):
    """q_alpha for the two-tailed Nemenyi test with k treatments."""
    if alpha == 0.05 and k in NEMENYI_Q_ALPHA_005:
        return NEMENYI_Q_ALPHA_005[k]
    q = studentized_range.ppf(1.0 - alpha, k, np.inf) / np.sqrt(2.0)
    return round(float(q), 3)


def _verify_nemenyi_fallback():
    """Check that the studentized-range fallback matches the Demsar table for k=2..10."""
    for k, table_val in NEMENYI_Q_ALPHA_005.items():
        q = studentized_range.ppf(0.95, k, np.inf) / np.sqrt(2.0)
        diff = abs(round(float(q), 3) - table_val)
        assert diff <= 0.002, f"Nemenyi fallback mismatch at k={k}: {q:.4f} vs table {table_val} (diff={diff})"


_verify_nemenyi_fallback()


def spearman_score(rank_scores, true_R_row):
    """Spearman rho between predicted rank_scores and the true rank row."""
    algos = list(true_R_row.index)
    pred = rank_scores.reindex(algos).values.astype(float)
    true = true_R_row.values.astype(float)
    rho, _ = spearmanr(pred, true)
    return float(rho)


def loss_curve(order, P_row):
    """loss(t) = best possible - best achieved among first t recommended, t=1..m."""
    best = P_row.max()
    running = -np.inf
    m = len(order)
    losses = np.empty(m, dtype=float)
    for t, algo in enumerate(order):
        running = max(running, P_row[algo])
        losses[t] = best - running
    return losses


def auc_loss_from_curve_df(loss_df):
    """Per-dataset AUC_loss = mean of the loss curve over t = 1..m (assignment definition)."""
    return loss_df.mean(axis=1)


def run_loo(build_fn, X, P, R,
            needs_X, dataset_ids=None, progress_cb=None):
    """LOO evaluation of one approach. Returns per-dataset Spearman, loss curves and recommended orders."""
    if dataset_ids is None:
        dataset_ids = list(P.index)
    m = P.shape[1]
    spearman_res = {}
    loss_res = {}
    orders = {}
    for i, did in enumerate(dataset_ids):
        if progress_cb is not None:
            progress_cb(i, did)
        train_ids = [d for d in dataset_ids if d != did]
        assert did not in train_ids  # anti-leakage: held-out id excluded from training slice
        X_train = X.loc[train_ids]
        P_train = P.loc[train_ids]
        R_train = R.loc[train_ids]

        model = build_fn()
        model.fit(X_train, P_train, R_train)

        X_query = X.loc[[did]].values if needs_X else None
        order, rank_scores = model.recommend(X_query)

        true_R_row = R.loc[did]
        spearman_res[did] = spearman_score(rank_scores, true_R_row)

        P_row = P.loc[did]
        loss_res[did] = loss_curve(order, P_row)
        orders[did] = order

    spearman_series = pd.Series(spearman_res, name="spearman")
    loss_df = pd.DataFrame.from_dict(
        loss_res, orient="index", columns=[f"t{t}" for t in range(1, m + 1)])
    loss_df = loss_df.loc[spearman_series.index]
    return {"per_dataset_spearman": spearman_series, "loss_curves": loss_df, "orders": orders}


def assert_no_leakage(X, P, R,
                       build_fn, held_out_id):
    """Fit without the held-out dataset and assert its id never reaches `fit`."""
    train_ids = [d for d in P.index if d != held_out_id]
    X_train = X.loc[train_ids]
    P_train = P.loc[train_ids]
    R_train = R.loc[train_ids]
    assert held_out_id not in X_train.index
    assert held_out_id not in P_train.index
    assert held_out_id not in R_train.index
    assert len(train_ids) == len(P.index) - 1
    model = build_fn()
    model.fit(X_train, P_train, R_train)
    return True


def mean_ranks(df, higher_is_better):
    """Mean rank per approach across datasets (rank 1 = best, ties averaged)."""
    if higher_is_better:
        ranks = np.apply_along_axis(lambda row: rankdata(-row, method="average"), 1, df.values)
    else:
        ranks = np.apply_along_axis(lambda row: rankdata(row, method="average"), 1, df.values)
    ranks = pd.DataFrame(ranks, index=df.index, columns=df.columns)
    return ranks.mean(axis=0)


def friedman_nemenyi(df, higher_is_better, alpha=0.05):
    """Friedman test plus Nemenyi critical difference and significantly different pairs."""
    k = df.shape[1]
    N = df.shape[0]
    samples = [df[c].values for c in df.columns]
    stat, p = friedmanchisquare(*samples)
    ranks = mean_ranks(df, higher_is_better)
    q_alpha = nemenyi_q_alpha(k, alpha)
    cd = q_alpha * np.sqrt(k * (k + 1) / (6.0 * N))

    approaches = list(df.columns)
    sig_pairs, nonsig_pairs = [], []
    for i in range(k):
        for j in range(i + 1, k):
            diff = abs(ranks[approaches[i]] - ranks[approaches[j]])
            entry = (approaches[i], approaches[j], float(diff))
            (sig_pairs if diff >= cd else nonsig_pairs).append(entry)

    return {
        "statistic": float(stat), "pvalue": float(p), "mean_ranks": ranks,
        "cd": float(cd), "q_alpha": q_alpha, "k": k, "N": N,
        "significant_pairs": sig_pairs, "nonsignificant_pairs": nonsig_pairs,
    }


def plot_cd_diagram(mean_ranks_, cd, title, out_path,
                     figsize=(9, 4.2), dpi=150,
                     tick_fontsize=9, label_fontsize=9,
                     cd_fontsize=9, title_fontsize=12,
                     tight=False):
    """Critical-difference diagram (Demsar 2006): mean-rank axis with crossbars for groups within CD."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ranks_sorted = mean_ranks_.sort_values()
    names = ranks_sorted.index.tolist()
    values = ranks_sorted.values
    k = len(names)
    lo, hi = 1, k

    # smaller labels for large k
    label_top = 0.92
    if k > 10:
        if label_fontsize == 9:
            label_fontsize = 7
        if tick_fontsize == 9:
            tick_fontsize = 8
        label_top = 0.98

    fig, ax = plt.subplots(figsize=figsize)
    ax.set_xlim(lo - 1.6, hi + 1.6)
    ax.set_ylim(0, 1.15)
    ax.axis("off")

    axis_y = 0.48
    ax.plot([lo, hi], [axis_y, axis_y], color="black", lw=1.3)
    for r in range(lo, hi + 1):
        ax.plot([r, r], [axis_y - 0.015, axis_y + 0.015], color="black", lw=1.3)
        ax.text(r, axis_y + 0.05, str(r), ha="center", va="bottom", fontsize=tick_fontsize)

    half = (k + 1) // 2
    left_names, right_names = names[:half], names[half:]

    n_left = max(len(left_names), 1)
    for idx, nm in enumerate(left_names):
        r = mean_ranks_[nm]
        y = axis_y + 0.14 + idx * ((label_top - axis_y - 0.16) / n_left)
        ax.plot([r, lo - 0.9], [axis_y, y], color="gray", lw=0.8)
        ax.text(lo - 1.0, y, f"{nm} ({r:.2f})", ha="right", va="center", fontsize=label_fontsize)

    n_right = max(len(right_names), 1)
    for idx, nm in enumerate(right_names):
        r = mean_ranks_[nm]
        y = axis_y - 0.15 - idx * ((axis_y - 0.15) / n_right)
        ax.plot([r, hi + 0.9], [axis_y, y], color="gray", lw=0.8)
        ax.text(hi + 1.0, y, f"{nm} ({r:.2f})", ha="left", va="center", fontsize=label_fontsize)

    # CD reference bar (top-left corner)
    cd_y = 1.05
    x0 = lo
    ax.plot([x0, x0 + cd], [cd_y, cd_y], color="black", lw=1.6)
    ax.plot([x0, x0], [cd_y - 0.02, cd_y + 0.02], color="black", lw=1.6)
    ax.plot([x0 + cd, x0 + cd], [cd_y - 0.02, cd_y + 0.02], color="black", lw=1.6)
    ax.text(x0 + cd / 2, cd_y + 0.025, f"CD = {cd:.3f}", ha="center", va="bottom",
            fontsize=cd_fontsize)

    # clique bars: connect consecutive (sorted-by-rank) approaches whose span < CD
    cliques = []
    i = 0
    while i < k:
        j = i
        while j + 1 < k and (values[j + 1] - values[i]) < cd:
            j += 1
        if j > i:
            cliques.append((i, j))
        i += 1
    cliques = [c for c in cliques
               if not any(o != c and o[0] <= c[0] and o[1] >= c[1] for o in cliques)]

    bar_y0 = axis_y - 0.06
    for idx, (i0, j0) in enumerate(cliques):
        y = bar_y0 - idx * 0.05
        ax.plot([values[i0], values[j0]], [y, y], color="black", lw=4, solid_capstyle="butt")

    ax.set_title(title, fontsize=title_fontsize, pad=20)
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight" if tight else None)
    plt.close(fig)
