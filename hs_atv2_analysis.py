"""Analysis of the new meta-features: relation to meta-targets, redundancy, filtered X and extraction cost.

Usage: python hs_atv2_analysis.py [--data-dir data_hs] [--results-dir results_atv2] [--quick]
"""

import argparse
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.cluster.hierarchy import dendrogram, fcluster, linkage
from scipy.spatial.distance import squareform
from scipy.stats import spearmanr
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.model_selection import KFold

import approaches as ap

ROOT = Path(__file__).resolve().parent
RANDOM_STATE = 0

SPARSE_ALGOS = ["tfidf_word_lr", "tfidf_char_svm", "tfidf_nb", "svd_rf", "svd_histgb", "svd_knn"]
DENSE_ALGOS = ["minilm_lr", "minilm_mlp", "hatebert_lr", "mbert_lr"]

REDUNDANCY_CUT_DIST = 0.3  # |rho| >= 0.7
DUP_THRESHOLD = 0.98

# extraction_cost groups -> catalog groups (shared_knn is shared ih_/mt_ overhead, no catalog group)
COST_GROUP_TO_CATALOG_GROUPS = {
    "base_balance": ["simple", "balance"],
    "base_text": ["lexical", "social_media"],
    "base_mutual_info": ["info_theory"],
    "base_embedding": ["separability"],
    "base_manifest": ["manifest"],
    "shared_knn": [],
}


def catalog_groups_for_cost_group(cost_group, catalog_groups_present):
    """Catalog group(s) covered by a cost-table group."""
    if cost_group in catalog_groups_present:
        return [cost_group]
    return COST_GROUP_TO_CATALOG_GROUPS.get(cost_group, [])


def parse_args():
    p = argparse.ArgumentParser(description="Atividade 2 -- analyses 1+2 (association, redundancy).")
    p.add_argument("--data-dir", default="data_hs")
    p.add_argument("--results-dir", default="results_atv2")
    p.add_argument("--n-jobs", type=int, default=10)
    p.add_argument("--quick", action="store_true",
                    help="Reduce RF estimators/seeds/repeats for a fast smoke test only. "
                         "NEVER use for a real run.")
    return p.parse_args()


def resolve(path_str):
    p = Path(path_str)
    return p if p.is_absolute() else (ROOT / p)


def load_data(data_dir):
    X = pd.read_csv(data_dir / "X.csv", index_col="dataset_id")
    X_new = pd.read_csv(data_dir / "X_new.csv", index_col="dataset_id")
    X_ext = pd.read_csv(data_dir / "X_extended.csv", index_col="dataset_id")
    P = pd.read_csv(data_dir / "P.csv", index_col="dataset_id")
    catalog = pd.read_csv(data_dir / "meta_feature_catalog.csv")
    cost = pd.read_csv(data_dir / "extraction_cost.csv")
    return X, X_new, X_ext, P, catalog, cost


def compute_targets(P):
    """Rank matrix R plus best_f1 and paradigm_gap targets; also returns best_algo separately."""
    R = ap.ranks_from_P(P)
    best_f1 = P.max(axis=1).rename("best_f1")
    paradigm_gap = (P[SPARSE_ALGOS].max(axis=1) - P[DENSE_ALGOS].max(axis=1)).rename("paradigm_gap")
    best_algo = P.idxmax(axis=1).rename("best_algo")
    targets = pd.concat([R, best_f1, paradigm_gap], axis=1)
    return targets, best_algo


def bh_fdr(pvals):
    """Benjamini-Hochberg FDR correction; returns q-values in input order."""
    n = len(pvals)
    order = np.argsort(pvals)
    ranked = pvals[order]
    q_ranked = ranked * n / (np.arange(1, n + 1))
    # enforce monotonicity from the largest p-value down
    q_ranked = np.minimum.accumulate(q_ranked[::-1])[::-1]
    q_ranked = np.clip(q_ranked, 0, 1)
    q = np.empty(n)
    q[order] = q_ranked
    return q


def association_analysis(X_ext, targets, catalog,
                          out_dir):
    cat = catalog.set_index("feature")
    rows = []
    for feat in X_ext.columns:
        origin = cat.loc[feat, "origin"] if feat in cat.index else "unknown"
        family = cat.loc[feat, "family"] if feat in cat.index else "unknown"
        x = X_ext[feat].values.astype(float)
        for tgt in targets.columns:
            y = targets[tgt].values.astype(float)
            if np.std(x) == 0 or np.std(y) == 0:
                rho, p = 0.0, 1.0
            else:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    rho, p = spearmanr(x, y)
                if np.isnan(rho):
                    rho, p = 0.0, 1.0
            rows.append({"feature": feat, "origin": origin, "family": family,
                         "target": tgt, "rho": float(rho), "p": float(p)})
    df = pd.DataFrame(rows)
    # BH-FDR within each origin block
    df["q"] = np.nan
    for origin, sub in df.groupby("origin"):
        df.loc[sub.index, "q"] = bh_fdr(sub["p"].values)
    df.to_csv(out_dir / "assoc_spearman.csv", index=False)

    print(f"\n=== Association analysis: {len(df)} (feature x target) tests, "
          f"{df['origin'].nunique()} origin block(s) BH-corrected separately ===")
    n_sig_new = int(((df.origin == "new") & (df.q < 0.05)).sum())
    n_new_tests = int((df.origin == "new").sum())
    print(f"new-feature tests with q<0.05: {n_sig_new}/{n_new_tests}")

    plot_assoc_heatmap(df, out_dir / "fig_assoc_heatmap.png")
    return df


def plot_assoc_heatmap(assoc_df, out_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    new_df = assoc_df[assoc_df.origin == "new"].copy()
    order = (new_df[["family", "feature"]].drop_duplicates()
             .sort_values(["family", "feature"]))
    features = order["feature"].tolist()
    families = order["family"].tolist()
    targets = list(assoc_df["target"].unique())

    pivot_rho = new_df.pivot(index="feature", columns="target", values="rho").reindex(
        index=features, columns=targets)
    pivot_q = new_df.pivot(index="feature", columns="target", values="q").reindex(
        index=features, columns=targets)

    fig_h = max(4.0, 0.18 * len(features) + 1.5)
    fig, ax = plt.subplots(figsize=(8.5, min(fig_h, 20)))
    im = ax.imshow(pivot_rho.values, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")

    for i in range(pivot_q.shape[0]):
        for j in range(pivot_q.shape[1]):
            if pd.notna(pivot_q.values[i, j]) and pivot_q.values[i, j] < 0.05:
                ax.plot(j, i, marker=".", color="black", markersize=3)

    ax.set_xticks(range(len(targets)))
    ax.set_xticklabels(targets, rotation=90, fontsize=6)
    ax.set_yticks(range(len(features)))
    ax.set_yticklabels(features, fontsize=5)

    prev = None
    for i, fam in enumerate(families):
        if fam != prev:
            ax.axhline(i - 0.5, color="black", lw=0.6)
            prev = fam
    # family label per contiguous block, on a secondary right-hand axis
    ax2 = ax.secondary_yaxis("right")
    block_starts, block_labels = [], []
    prev = None
    for i, fam in enumerate(families):
        if fam != prev:
            block_starts.append(i)
            block_labels.append(fam)
            prev = fam
    mids = [ (s + (block_starts[k + 1] if k + 1 < len(block_starts) else len(features))) / 2.0 - 0.5
             for k, s in enumerate(block_starts) ]
    ax2.set_yticks(mids)
    ax2.set_yticklabels(block_labels, fontsize=5)

    fig.colorbar(im, ax=ax, shrink=0.4, label="rho de Spearman", pad=0.22, fraction=0.05)
    ax.set_title("Novas meta-features vs meta-alvos (ponto: q<0,05)", fontsize=9)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _impurity_importance_one_seed(X_arr, y_arr, seed, n_estimators, min_samples_leaf):
    rf = RandomForestRegressor(n_estimators=n_estimators, min_samples_leaf=min_samples_leaf,
                                random_state=seed)
    rf.fit(X_arr, y_arr)
    return rf.feature_importances_


def _perm_importance_one_seed(X_arr, y_arr, seed, n_estimators, min_samples_leaf,
                               n_splits, n_repeats):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    fold_imps = []
    for train_idx, test_idx in kf.split(X_arr):
        rf = RandomForestRegressor(n_estimators=n_estimators, min_samples_leaf=min_samples_leaf,
                                    random_state=seed)
        rf.fit(X_arr[train_idx], y_arr[train_idx])
        res = permutation_importance(rf, X_arr[test_idx], y_arr[test_idx],
                                      scoring="neg_mean_absolute_error",
                                      n_repeats=n_repeats, random_state=seed)
        fold_imps.append(res.importances_mean)
    return np.mean(fold_imps, axis=0)


def importance_for_target(X_ext, y, seeds, n_estimators,
                           min_samples_leaf, n_splits, n_repeats, n_jobs):
    X_arr = np.asarray(X_ext.values, dtype=float)
    y_arr = np.asarray(y.values, dtype=float)
    impurity_list = Parallel(n_jobs=n_jobs)(
        delayed(_impurity_importance_one_seed)(X_arr, y_arr, s, n_estimators, min_samples_leaf)
        for s in seeds)
    perm_list = Parallel(n_jobs=n_jobs)(
        delayed(_perm_importance_one_seed)(X_arr, y_arr, s, n_estimators, min_samples_leaf,
                                            n_splits, n_repeats)
        for s in seeds)
    return np.mean(impurity_list, axis=0), np.mean(perm_list, axis=0)


def importance_analysis(X_ext, targets, catalog,
                         out_dir, n_jobs, quick):
    n_estimators = 50 if quick else 500
    min_samples_leaf = 2
    n_seeds = 2 if quick else 10
    n_splits = 3 if quick else 5
    n_repeats = 2 if quick else 5
    seeds = list(range(n_seeds))

    cat = catalog.set_index("feature")
    all_algorithm_cols = list(targets.columns)  # 10 rank cols + best_f1 + paradigm_gap
    features = list(X_ext.columns)

    rows = []
    print(f"\n=== Importance analysis: {len(all_algorithm_cols)} targets x "
          f"{len(seeds)} seeds x (impurity RF + {n_splits}-fold perm importance) ===")
    for tgt in all_algorithm_cols:
        print(f"  [{tgt}] fitting...", flush=True)
        imp_impurity, imp_perm = importance_for_target(
            X_ext, targets[tgt], seeds, n_estimators, min_samples_leaf, n_splits, n_repeats, n_jobs)
        for feat, vi, vp in zip(features, imp_impurity, imp_perm):
            rows.append({
                "feature": feat,
                "origin": cat.loc[feat, "origin"] if feat in cat.index else "unknown",
                "family": cat.loc[feat, "family"] if feat in cat.index else "unknown",
                "algorithm": tgt,
                "imp_impurity": float(vi),
                "imp_perm": float(vp),
            })
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "importance_by_algo.csv", index=False)

    print("\nTop-10 features per target (by impurity importance):")
    for tgt in all_algorithm_cols:
        top = df[df.algorithm == tgt].sort_values("imp_impurity", ascending=False).head(10)
        print(f"  {tgt}: " + ", ".join(f"{r.feature}({r.imp_impurity:.3f})" for r in top.itertuples()))

    # family aggregation
    fam_rows = []
    for tgt in all_algorithm_cols:
        sub = df[df.algorithm == tgt]
        total = sub["imp_impurity"].sum()
        new_total = sub[sub.origin == "new"]["imp_impurity"].sum()
        for family, fsub in sub.groupby("family"):
            fam_rows.append({
                "family": family,
                "origin": fsub["origin"].iloc[0],
                "algorithm": tgt,
                "imp_impurity_sum": float(fsub["imp_impurity"].sum()),
                "share_of_total": float(fsub["imp_impurity"].sum() / total) if total > 0 else 0.0,
            })
        fam_rows.append({
            "family": "__TOTAL_NEW_SHARE__", "origin": "new", "algorithm": tgt,
            "imp_impurity_sum": float(new_total),
            "share_of_total": float(new_total / total) if total > 0 else 0.0,
        })
    fam_df = pd.DataFrame(fam_rows)
    fam_df.to_csv(out_dir / "importance_family_by_algo.csv", index=False)

    print("\nShare of total impurity importance from origin=new, per algorithm/target:")
    new_share = fam_df[fam_df.family == "__TOTAL_NEW_SHARE__"].set_index("algorithm")["share_of_total"]
    print(new_share.round(4).to_string())

    plot_importance_family_heatmap(fam_df, out_dir / "fig_importance_family_by_algo.png")
    return df, fam_df


def plot_importance_family_heatmap(fam_df, out_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plot_df = fam_df[fam_df.family != "__TOTAL_NEW_SHARE__"]
    pivot = plot_df.pivot_table(index="family", columns="algorithm", values="share_of_total",
                                 aggfunc="sum").fillna(0.0)
    fig, ax = plt.subplots(figsize=(max(6, 0.6 * pivot.shape[1]), max(4, 0.35 * pivot.shape[0]) + 1))
    im = ax.imshow(pivot.values * 100, cmap="viridis", aspect="auto")
    ax.set_xticks(range(pivot.shape[1]))
    ax.set_xticklabels(pivot.columns, rotation=90, fontsize=7)
    ax.set_yticks(range(pivot.shape[0]))
    ax.set_yticklabels(pivot.index, fontsize=7)
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            val = pivot.values[i, j] * 100
            ax.text(j, i, f"{val:.0f}%", ha="center", va="center", fontsize=6,
                    color="white" if val > pivot.values.max() * 100 * 0.5 else "black")
    fig.colorbar(im, ax=ax, shrink=0.7, label="% da importância (impureza)")
    ax.set_title("Participação da importância por família, por algoritmo/alvo", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def redundancy_analysis(X_ext, catalog, cost,
                         out_dir):
    cat = catalog.set_index("feature")
    corr = X_ext.corr(method="spearman").abs()
    features = list(X_ext.columns)

    # per-new-feature max correlations
    rows = []
    for feat in features:
        origin = cat.loc[feat, "origin"] if feat in cat.index else "unknown"
        if origin != "new":
            continue
        base_feats = [f for f in features if (f in cat.index and cat.loc[f, "origin"] == "base")]
        new_feats = [f for f in features if (f in cat.index and cat.loc[f, "origin"] == "new" and f != feat)]
        max_base = corr.loc[feat, base_feats].idxmax() if base_feats else None
        max_base_val = corr.loc[feat, base_feats].max() if base_feats else np.nan
        max_new_val = corr.loc[feat, new_feats].max() if new_feats else np.nan
        rows.append({"feature": feat, "family": cat.loc[feat, "family"],
                     "max_abs_rho_base": float(max_base_val) if pd.notna(max_base_val) else np.nan,
                     "max_abs_rho_base_feature": max_base,
                     "max_abs_rho_new": float(max_new_val) if pd.notna(max_new_val) else np.nan})
    new_corr_df = pd.DataFrame(rows)
    new_corr_df.to_csv(out_dir / "new_feature_max_corr.csv", index=False)

    # hierarchical clustering
    dist = 1.0 - corr.values
    dist = np.clip(dist, 0, None)
    np.fill_diagonal(dist, 0.0)
    dist = (dist + dist.T) / 2.0  # enforce exact symmetry against fp noise
    condensed = squareform(dist, checks=False)
    Z = linkage(condensed, method="average")
    cluster_ids = fcluster(Z, t=REDUNDANCY_CUT_DIST, criterion="distance")

    cluster_rows = []
    for feat, cid in zip(features, cluster_ids):
        origin = cat.loc[feat, "origin"] if feat in cat.index else "unknown"
        family = cat.loc[feat, "family"] if feat in cat.index else "unknown"
        cluster_rows.append({"cluster_id": int(cid), "feature": feat, "origin": origin, "family": family})
    clusters_df = pd.DataFrame(cluster_rows)

    comp_map = {}
    for cid, sub in clusters_df.groupby("cluster_id"):
        origins = set(sub["origin"])
        if origins == {"base"}:
            comp_map[cid] = "base_only"
        elif origins == {"new"}:
            comp_map[cid] = "new_only"
        else:
            comp_map[cid] = "mixed"
    clusters_df["composition"] = clusters_df["cluster_id"].map(comp_map)
    clusters_df.to_csv(out_dir / "redundancy_clusters.csv", index=False)

    n_clusters = clusters_df["cluster_id"].nunique()
    comp_counts = clusters_df.drop_duplicates("cluster_id")["composition"].value_counts()
    frac_ge_07 = float((new_corr_df["max_abs_rho_base"] >= 0.7).mean()) if len(new_corr_df) else float("nan")
    frac_ge_09 = float((new_corr_df["max_abs_rho_base"] >= 0.9).mean()) if len(new_corr_df) else float("nan")
    summary = {
        "n_clusters": int(n_clusters),
        "n_base_only": int(comp_counts.get("base_only", 0)),
        "n_new_only": int(comp_counts.get("new_only", 0)),
        "n_mixed": int(comp_counts.get("mixed", 0)),
        "frac_new_features_max_rho_base_ge_0.7": frac_ge_07,
        "frac_new_features_max_rho_base_ge_0.9": frac_ge_09,
    }
    (out_dir / "redundancy_summary.json").write_text(json.dumps(summary, indent=2))
    print("\n=== Redundancy summary ===")
    print(json.dumps(summary, indent=2))

    plot_dendrogram(Z, features, cat, REDUNDANCY_CUT_DIST, out_dir / "fig_redundancy_dendrogram.png")
    plot_corr_new_vs_base(corr, cat, out_dir / "fig_corr_new_vs_base.png")

    return corr, clusters_df, cluster_ids, summary


FAMILY_COLOR_CYCLE = ["#e6194b", "#3cb44b", "#4363d8", "#f58231", "#911eb4",
                       "#46f0f0", "#f032e6", "#bcf60c", "#008080", "#9a6324",
                       "#800000", "#808000"]


def plot_dendrogram(Z, features, cat, cut_dist, out_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    families = sorted({cat.loc[f, "family"] for f in features if f in cat.index and cat.loc[f, "origin"] == "new"})
    fam_color = {fam: FAMILY_COLOR_CYCLE[i % len(FAMILY_COLOR_CYCLE)] for i, fam in enumerate(families)}

    fig, ax = plt.subplots(figsize=(7, max(6, 0.16 * len(features) + 2)))
    dn = dendrogram(Z, labels=features, orientation="left", ax=ax, color_threshold=cut_dist,
                     above_threshold_color="lightgray")
    ax.axvline(cut_dist, color="red", linestyle="--", lw=1.0, label=f"corte = {cut_dist} (|rho|>=0,7)")
    for lbl in ax.get_ymajorticklabels():
        feat = lbl.get_text()
        if feat in cat.index and cat.loc[feat, "origin"] == "new":
            fam = cat.loc[feat, "family"]
            lbl.set_color(fam_color.get(fam, "black"))
        else:
            lbl.set_color("black")
        lbl.set_fontsize(5)
    ax.set_xlabel("distância (1 − |rho de Spearman|)")
    ax.set_title("Agrupamento hierárquico das features de X_extended", fontsize=9)
    ax.legend(fontsize=7, loc="lower right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_corr_new_vs_base(corr, cat, out_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy.cluster.hierarchy import leaves_list

    base_feats = [f for f in corr.columns if f in cat.index and cat.loc[f, "origin"] == "base"]
    new_feats = [f for f in corr.columns if f in cat.index and cat.loc[f, "origin"] == "new"]
    if not base_feats or not new_feats:
        return

    def cluster_order(feat_list):
        sub = corr.loc[feat_list, feat_list]
        dist = np.clip(1.0 - sub.values, 0, None)
        np.fill_diagonal(dist, 0.0)
        dist = (dist + dist.T) / 2.0
        Z = linkage(squareform(dist, checks=False), method="average")
        order = leaves_list(Z)
        return [feat_list[i] for i in order]

    base_order = cluster_order(base_feats)
    new_order = cluster_order(new_feats)
    sub = corr.loc[new_order, base_order]

    fig, ax = plt.subplots(figsize=(max(7, 0.25 * len(base_order)), max(6, 0.2 * len(new_order) + 1)))
    im = ax.imshow(sub.values, cmap="magma", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(base_order)))
    ax.set_xticklabels(base_order, rotation=90, fontsize=5)
    ax.set_yticks(range(len(new_order)))
    ax.set_yticklabels(new_order, fontsize=5)
    fig.colorbar(im, ax=ax, shrink=0.7, label="|rho de Spearman|")
    ax.set_title("Correlação entre features novas e base (ordem agrupada)", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def build_filtered_X(X_ext, catalog, cost,
                      corr, clusters_df, out_dir, data_dir):
    cat = catalog.set_index("feature")
    mean_cost_per_group = cost.groupby("group")["seconds"].mean()
    group_of = cat["group"] if "group" in cat.columns else pd.Series(dtype=object)

    selection_rows = []
    kept_features = []

    for cid, sub in clusters_df.groupby("cluster_id"):
        members = sub["feature"].tolist()
        if len(members) == 1:
            rep = members[0]
            reason = "singleton_cluster"
        else:
            base_members = sub[sub.origin == "base"]["feature"].tolist()
            if base_members:
                mean_rho = {f: corr.loc[f, members].drop(f).mean() for f in base_members}
                rep = max(mean_rho, key=mean_rho.get)
                reason = "cluster_representative_base_medoid"
            else:
                new_members = members
                costs = {f: mean_cost_per_group.get(group_of.get(f, None), np.inf) for f in new_members}
                min_cost = min(costs.values())
                tied = [f for f, c in costs.items() if c == min_cost]
                if len(tied) > 1:
                    mean_rho = {f: corr.loc[f, members].drop(f).mean() for f in tied}
                    rep = max(mean_rho, key=mean_rho.get)
                else:
                    rep = tied[0]
                reason = "cluster_representative_new_lowest_cost"
        for f in members:
            selection_rows.append({
                "cluster_id": int(cid), "feature": f,
                "origin": cat.loc[f, "origin"] if f in cat.index else "unknown",
                "family": cat.loc[f, "family"] if f in cat.index else "unknown",
                "kept": f == rep,
                "reason": reason if f == rep else "dropped_redundant_in_cluster",
            })
        kept_features.append(rep)

    # secondary pass: drop near-duplicates (|rho| >= 0.98) among the kept set
    final_kept = []
    for f in kept_features:
        dup_of = None
        for g in final_kept:
            if corr.loc[f, g] >= DUP_THRESHOLD:
                dup_of = g
                break
        if dup_of is None:
            final_kept.append(f)
        else:
            for row in selection_rows:
                if row["feature"] == f and row["kept"]:
                    row["kept"] = False
                    row["reason"] = f"dropped_duplicate_ge_{DUP_THRESHOLD}_of_{dup_of}"

    sel_df = pd.DataFrame(selection_rows)
    sel_df.to_csv(out_dir / "filtered_selection.csv", index=False)

    X_filtered = X_ext[final_kept].copy()
    X_filtered.to_csv(data_dir / "X_ext_filtered.csv")

    print(f"\n=== Unsupervised filtered X: kept {len(final_kept)}/{X_ext.shape[1]} features ===")
    print(f"Written to {data_dir / 'X_ext_filtered.csv'}")
    return X_filtered, sel_df


def cost_analysis(catalog, cost, out_dir):
    n_datasets = cost["dataset_id"].nunique()
    catalog_groups_present = set(catalog["group"].unique())
    n_features_per_group = catalog.groupby("group").size()
    family_per_group = catalog.groupby("group")["family"].first()
    origin_per_group = catalog.groupby("group")["origin"].first()

    rows = []
    for cost_group, sub in cost.groupby("group"):
        total_seconds = float(sub["seconds"].sum())
        mapped = catalog_groups_for_cost_group(cost_group, catalog_groups_present)
        n_feat = int(sum(n_features_per_group.get(g, 0) for g in mapped))
        families = sorted({family_per_group[g] for g in mapped if g in family_per_group.index})
        origins = {origin_per_group[g] for g in mapped if g in origin_per_group.index}
        if not origins:
            origin = "base" if cost_group.startswith("base_") else "new"
        else:
            origin = origins.pop() if len(origins) == 1 else "+".join(sorted(origins))
        family = "+".join(families) if families else (
            "pre-processamento compartilhado (ih/mt)" if cost_group == "shared_knn" else "unknown")
        mean_per_dataset = total_seconds / n_datasets
        rows.append({
            "group": cost_group,
            "family": family,
            "origin": origin,
            "n_features": n_feat,
            "total_seconds": total_seconds,
            "mean_seconds_per_dataset": mean_per_dataset,
            "seconds_per_feature": mean_per_dataset / n_feat if n_feat else np.nan,
        })
    df = pd.DataFrame(rows).sort_values("total_seconds", ascending=False)
    df.to_csv(out_dir / "cost_by_group.csv", index=False)
    print("\n=== Cost by group ===")
    print(df.round(4).to_string(index=False))
    return df


def main():
    args = parse_args()
    data_dir = resolve(args.data_dir)
    results_dir = resolve(args.results_dir)
    out_dir = results_dir / "analysis"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"=== Loading data from {data_dir} ===", flush=True)
    X, X_new, X_ext, P, catalog, cost = load_data(data_dir)
    print(f"X {X.shape}, X_new {X_new.shape}, X_extended {X_ext.shape}, P {P.shape}, "
          f"catalog {catalog.shape}, cost {cost.shape}", flush=True)

    targets, best_algo = compute_targets(P)
    print(f"Meta-targets: {list(targets.columns)}; best_algo value counts:")
    print(best_algo.value_counts().to_string())

    print("\n=== (b) Association analysis ===", flush=True)
    assoc_df = association_analysis(X_ext, targets, catalog, out_dir)

    print("\n=== (c) Importance per algorithm ===", flush=True)
    imp_df, fam_imp_df = importance_analysis(X_ext, targets, catalog, out_dir,
                                              n_jobs=args.n_jobs, quick=args.quick)

    print("\n=== (d) Redundancy analysis ===", flush=True)
    corr, clusters_df, cluster_ids, redundancy_summary = redundancy_analysis(X_ext, catalog, cost, out_dir)

    print("\n=== (e) Unsupervised filtered X ===", flush=True)
    X_filtered, sel_df = build_filtered_X(X_ext, catalog, cost, corr, clusters_df, out_dir, data_dir)

    print("\n=== (f) Cost analysis ===", flush=True)
    cost_df = cost_analysis(catalog, cost, out_dir)

    print(f"\n=== Done. Files written to {out_dir} ===")
    print(sorted(p.name for p in out_dir.iterdir()))


if __name__ == "__main__":
    main()
