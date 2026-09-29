"""Effect of the new meta-features on the recommenders across X variants, under LODO and LOSO.

Usage: python hs_atv2_compare.py [--data-dir data_hs] [--results-dir results_atv2]
       [--protocols lodo,loso] [--n-jobs 10] [--quick]
"""

import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.stats import wilcoxon
from sklearn.ensemble import RandomForestRegressor

import approaches as ap
import evaluate as ev
from hs_leave_one_source_out import build_source_groups, run_loso
from run_experiments import _df_to_md_table

ROOT = Path(__file__).resolve().parent
RANDOM_STATE = 0
LAM = 0.5
K_SEL = 36  # matches n_features of the "base" variant (X.csv)

APPROACH_CODES = ["RegP", "RegR", "Harris"]
VARIANTS = ["base", "ext", "filt", "sel", "new", "cheap"]
CD_VARIANTS = list(VARIANTS)  # all variants -> 1 + 3*6 = 19 treatments in the CD/Friedman set
CHEAP_PREFIXES = ("tok_", "lm_", "lc_")


def parse_args():
    p = argparse.ArgumentParser(description="Atividade 2 -- analysis 3 (recommender comparison).")
    p.add_argument("--data-dir", default="data_hs")
    p.add_argument("--results-dir", default="results_atv2")
    p.add_argument("--protocols", default="lodo,loso")
    p.add_argument("--n-jobs", type=int, default=10)
    p.add_argument("--quick", action="store_true",
                    help="Reduce the SelectedFeatures RF estimators for a fast smoke test only. "
                         "NEVER use for a real run.")
    return p.parse_args()


def resolve(path_str):
    p = Path(path_str)
    return p if p.is_absolute() else (ROOT / p)


class SelectedFeatures(ap.BaseApproach):
    """Fit an inner approach on the top-k features chosen inside the training fold (RF importance)."""
    needs_X = True

    def __init__(self, inner_builder, k, n_estimators=500, random_state=RANDOM_STATE):
        self.inner_builder = inner_builder
        self.k = k
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.name = "SelectedFeatures"

    def fit(self, X_train, P_train, R_train, folds_train=None):
        rf = RandomForestRegressor(n_estimators=self.n_estimators, random_state=self.random_state)
        rf.fit(np.asarray(X_train.values, dtype=float), R_train.values)
        k = min(self.k, X_train.shape[1])
        self.selected_idx_ = np.argsort(-rf.feature_importances_)[:k]
        self.selected_cols_ = list(X_train.columns[self.selected_idx_])
        X_train_sel = X_train.iloc[:, self.selected_idx_]
        self.inner_ = self.inner_builder()
        self.inner_.fit(X_train_sel, P_train, R_train, folds_train)
        return self

    def recommend(self, X_query):
        X_query = np.asarray(X_query, dtype=float).reshape(1, -1)
        return self.inner_.recommend(X_query[:, self.selected_idx_])


def make_base_builder(approach_code):
    if approach_code == "RegP":
        return lambda: ap.RegressorOnP(random_state=RANDOM_STATE)
    if approach_code == "RegR":
        return lambda: ap.RegressorOnR(random_state=RANDOM_STATE)
    if approach_code == "Harris":
        return lambda: ap.Harris(lam=LAM, n_estimators=100, random_state=RANDOM_STATE, n_jobs=1)
    raise ValueError(f"unknown approach code {approach_code!r}")


def make_builder(approach_code, variant, quick):
    base_builder = make_base_builder(approach_code)
    if variant == "sel":
        n_est = 50 if quick else 500
        return (lambda: SelectedFeatures(base_builder, k=K_SEL, n_estimators=n_est)), True
    return base_builder, True


def load_variants(data_dir, dataset_ids):
    X = pd.read_csv(data_dir / "X.csv", index_col="dataset_id").loc[dataset_ids]
    X_new = pd.read_csv(data_dir / "X_new.csv", index_col="dataset_id").loc[dataset_ids]
    X_ext = pd.read_csv(data_dir / "X_extended.csv", index_col="dataset_id").loc[dataset_ids]
    filt_path = data_dir / "X_ext_filtered.csv"
    if not filt_path.exists():
        raise FileNotFoundError(
            f"{filt_path} not found -- run hs_atv2_analysis.py first (it writes X_ext_filtered.csv "
            f"into --data-dir).")
    X_filt = pd.read_csv(filt_path, index_col="dataset_id").loc[dataset_ids]
    cheap_cols = [c for c in X_new.columns if c.startswith(CHEAP_PREFIXES)]
    X_cheap = X_new[cheap_cols]
    return {"base": X, "ext": X_ext, "filt": X_filt, "sel": X_ext, "new": X_new, "cheap": X_cheap}


def n_features_for(variant, variant_X):
    if variant == "sel":
        return K_SEL
    return variant_X[variant].shape[1]


def run_one_treatment(name, approach_code, variant, builder, needs_X, protocol,
                       X_variant, P, R, dataset_ids, group_of=None):
    if protocol == "lodo":
        res = ev.run_loo(builder, X_variant, P, R, needs_X=needs_X, dataset_ids=dataset_ids)
    elif protocol == "loso":
        res = run_loso(builder, X_variant, P, R, needs_X=needs_X, group_of=group_of,
                        dataset_ids=dataset_ids)
    else:
        raise ValueError(f"unknown protocol {protocol!r}")
    return name, res


def build_treatment_list(quick, variant_X):
    """List of (name, approach_code, variant, builder, needs_X)."""
    treatments = [("AR", "AR", "AR", (lambda: ap.AverageRank()), False)]
    for approach_code in APPROACH_CODES:
        for variant in VARIANTS:
            builder, needs_X = make_builder(approach_code, variant, quick)
            treatments.append((f"{approach_code}|{variant}", approach_code, variant, builder, needs_X))
    return treatments


def run_protocol(protocol, treatments, variant_X, P, R, dataset_ids,
                  group_of, n_jobs):
    print(f"\n=== Running protocol={protocol}: {len(treatments)} treatments (n_jobs={n_jobs}) ===",
          flush=True)
    jobs = []
    for name, approach_code, variant, builder, needs_X in treatments:
        X_variant = variant_X[variant] if variant != "AR" else variant_X["base"]
        jobs.append(delayed(run_one_treatment)(
            name, approach_code, variant, builder, needs_X, protocol, X_variant, P, R,
            dataset_ids, group_of))
    results = Parallel(n_jobs=n_jobs)(jobs)
    return dict(results)


def assemble_tables(treatments, all_results, T_STEPS):
    names = [t[0] for t in treatments]
    per_dataset_spearman = pd.DataFrame({n: all_results[n]["per_dataset_spearman"] for n in names})
    per_dataset_spearman.index.name = "dataset_id"
    per_dataset_auc_loss = pd.DataFrame(
        {n: ev.auc_loss_from_curve_df(all_results[n]["loss_curves"]) for n in names})
    per_dataset_auc_loss.index.name = "dataset_id"
    mean_loss_curves = pd.DataFrame(
        {n: all_results[n]["loss_curves"].mean(axis=0).values for n in names},
        index=[f"t{t}" for t in range(1, T_STEPS + 1)]).T
    mean_loss_curves.index.name = "treatment"
    return per_dataset_spearman, per_dataset_auc_loss, mean_loss_curves


def build_summary(treatments, per_dataset_spearman, per_dataset_auc_loss, mean_loss_curves,
                   variant_X):
    rows = []
    for name, approach_code, variant, _, _ in treatments:
        rows.append({
            "treatment": name,
            "approach": approach_code,
            "variant": variant,
            "n_features": n_features_for(variant, variant_X) if variant != "AR" else 0,
            "mean_spearman": per_dataset_spearman[name].mean(),
            "std_spearman": per_dataset_spearman[name].std(),
            "mean_auc_loss": per_dataset_auc_loss[name].mean(),
            "std_auc_loss": per_dataset_auc_loss[name].std(),
            "loss_t1": mean_loss_curves.loc[name, "t1"],
        })
    return pd.DataFrame(rows)


def wilcoxon_vs_base(per_dataset_spearman, per_dataset_auc_loss):
    rows = []
    for approach_code in APPROACH_CODES:
        base_sp = per_dataset_spearman[f"{approach_code}|base"]
        base_auc = per_dataset_auc_loss[f"{approach_code}|base"]
        for variant in [v for v in VARIANTS if v != "base"]:
            name = f"{approach_code}|{variant}"
            diff_sp = per_dataset_spearman[name] - base_sp
            diff_auc = per_dataset_auc_loss[name] - base_auc
            wins = int((diff_sp > 0).sum())
            ties = int((diff_sp == 0).sum())
            losses = int((diff_sp < 0).sum())

            def safe_wilcoxon(diff):
                if np.allclose(diff.values, 0.0):
                    return float("nan")
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    try:
                        _, p = wilcoxon(diff.values, zero_method="wilcox")
                        return float(p)
                    except ValueError:
                        return float("nan")

            rows.append({
                "approach": approach_code, "variant": variant,
                "mean_diff_spearman": float(diff_sp.mean()),
                "mean_diff_auc_loss": float(diff_auc.mean()),
                "wins_spearman": wins, "ties_spearman": ties, "losses_spearman": losses,
                "wilcoxon_p_spearman": safe_wilcoxon(diff_sp),
                "wilcoxon_p_auc_loss": safe_wilcoxon(diff_auc),
            })
    return pd.DataFrame(rows)


def language_split(per_dataset_spearman, X_base, treatments):
    en_mask = X_base["is_english"].reindex(per_dataset_spearman.index) == 1
    rows = []
    for name, *_ in treatments:
        sp = per_dataset_spearman[name]
        rows.append({
            "treatment": name,
            "mean_spearman_english": float(sp[en_mask].mean()),
            "mean_spearman_non_english": float(sp[~en_mask].mean()),
            "n_english": int(en_mask.sum()),
            "n_non_english": int((~en_mask).sum()),
        })
    return pd.DataFrame(rows)


def friedman_nemenyi_block(per_dataset_spearman, per_dataset_auc_loss, out_dir):
    cd_names = ["AR"] + [f"{a}|{v}" for a in APPROACH_CODES for v in CD_VARIANTS]
    fr_sp = ev.friedman_nemenyi(per_dataset_spearman[cd_names], higher_is_better=True)
    fr_auc = ev.friedman_nemenyi(per_dataset_auc_loss[cd_names], higher_is_better=False)

    ev.plot_cd_diagram(fr_sp["mean_ranks"], fr_sp["cd"],
                        f"CD diagram -- mean Spearman rank (k={len(cd_names)}, CD={fr_sp['cd']:.3f})",
                        str(out_dir / "cd_diagram_spearman.png"), figsize=(10, 4), tight=True)
    ev.plot_cd_diagram(fr_auc["mean_ranks"], fr_auc["cd"],
                        f"CD diagram -- mean AUC_loss rank (k={len(cd_names)}, CD={fr_auc['cd']:.3f})",
                        str(out_dir / "cd_diagram_aucloss.png"), figsize=(10, 4), tight=True)

    lines = ["Friedman + Nemenyi post-hoc analysis -- Atividade 2 (19 treatments)",
             "=" * 70,
             f"k = {len(cd_names)} treatments = {cd_names}",
             ""]
    for metric_name, fr in [("Spearman (higher = better)", fr_sp), ("AUC_loss (lower = better)", fr_auc)]:
        lines.append(f"--- Metric: {metric_name} ---")
        lines.append(f"Friedman statistic = {fr['statistic']:.4f}, p-value = {fr['pvalue']:.6g}")
        lines.append(f"Nemenyi q_alpha(k={fr['k']}, alpha=0.05) = {fr['q_alpha']}")
        lines.append(f"CD = {fr['cd']:.4f}")
        lines.append("Mean ranks (rank 1 = best):")
        for n, r in fr["mean_ranks"].sort_values().items():
            lines.append(f"  {n:16s} {r:.3f}")
        lines.append("Pairs with |mean rank diff| >= CD (significantly different):")
        if fr["significant_pairs"]:
            for a, b, d in fr["significant_pairs"]:
                lines.append(f"  {a} vs {b}: diff = {d:.3f}")
        else:
            lines.append("  (none)")
        lines.append("")
    (out_dir / "friedman_nemenyi.txt").write_text("\n".join(lines) + "\n")
    return fr_sp, fr_auc


def plot_delta_spearman(per_dataset_spearman, X_base, out_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    en_mask = X_base["is_english"].reindex(per_dataset_spearman.index) == 1
    variants = [v for v in VARIANTS if v != "base"]
    fig, axes = plt.subplots(1, len(APPROACH_CODES), figsize=(4 * len(APPROACH_CODES), 4.5),
                              sharey=True)
    rng = np.random.default_rng(0)
    for ax, approach_code in zip(axes, APPROACH_CODES):
        base_sp = per_dataset_spearman[f"{approach_code}|base"]
        for i, variant in enumerate(variants):
            diff = per_dataset_spearman[f"{approach_code}|{variant}"] - base_sp
            jitter = rng.uniform(-0.12, 0.12, size=len(diff))
            colors = np.where(en_mask.values, "#4363d8", "#e6194b")
            ax.scatter(np.full(len(diff), i) + jitter, diff.values, c=colors, s=18, alpha=0.75,
                       edgecolors="none")
            ax.hlines(diff.mean(), i - 0.25, i + 0.25, color="black", lw=1.5)
        ax.axhline(0, color="gray", lw=0.8, linestyle="--")
        ax.set_xticks(range(len(variants)))
        ax.set_xticklabels(variants)
        ax.set_title(approach_code, fontsize=10)
    axes[0].set_ylabel("Spearman por dataset (variante − base)")
    handles = [plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="#4363d8", label="inglês", markersize=6),
               plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="#e6194b", label="não-inglês", markersize=6)]
    fig.legend(handles=handles, loc="upper center", ncol=2, fontsize=8, bbox_to_anchor=(0.5, 1.02))
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_loss_curves(mean_loss_curves, T_STEPS, out_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ts = np.arange(1, T_STEPS + 1)
    fig, axes = plt.subplots(1, len(APPROACH_CODES), figsize=(4 * len(APPROACH_CODES), 3), sharey=True)
    for ax, approach_code in zip(axes, APPROACH_CODES):
        for variant in VARIANTS:
            name = f"{approach_code}|{variant}"
            ax.plot(ts, mean_loss_curves.loc[name].values, marker="o", markersize=3, label=variant)
        ax.plot(ts, mean_loss_curves.loc["AR"].values, linestyle="--", color="gray", label="AR")
        ax.set_title(approach_code, fontsize=10)
        ax.set_xlabel("nº de pipelines testados $t$")
        ax.set_xticks(ts)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7)
    axes[0].set_ylabel("perda média (pontos de macro-F1)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def write_summary_md(out_dir, protocol, summary_df, wilcoxon_df, lang_df,
                      fr_sp, fr_auc, n_datasets):
    md = [f"# Atividade 2 -- comparison summary ({protocol})\n",
          f"N = {n_datasets} datasets.\n",
          "## Per-treatment summary\n",
          _df_to_md_table(summary_df.round(4)),
          "\n\n## Wilcoxon vs base (X-dependent approaches, variant != base)\n",
          _df_to_md_table(wilcoxon_df.round(4)),
          "\n\n## English vs non-English mean Spearman\n",
          _df_to_md_table(lang_df.round(4)),
          "\n\n## Friedman + Nemenyi -- Spearman (19 treatments)\n",
          f"statistic = {fr_sp['statistic']:.4f}, p = {fr_sp['pvalue']:.6g}, CD = {fr_sp['cd']:.4f}\n",
          "Mean ranks: " + ", ".join(f"{n}={r:.3f}" for n, r in fr_sp["mean_ranks"].sort_values().items()) + "\n",
          "\n\n## Friedman + Nemenyi -- AUC_loss (19 treatments)\n",
          f"statistic = {fr_auc['statistic']:.4f}, p = {fr_auc['pvalue']:.6g}, CD = {fr_auc['cd']:.4f}\n",
          "Mean ranks: " + ", ".join(f"{n}={r:.3f}" for n, r in fr_auc["mean_ranks"].sort_values().items()) + "\n",
          ]
    (out_dir / "summary.md").write_text("".join(md))


def main():
    args = parse_args()
    data_dir = resolve(args.data_dir)
    results_dir = resolve(args.results_dir)
    compare_dir = results_dir / "compare"
    protocols = [p.strip() for p in args.protocols.split(",") if p.strip()]

    print(f"=== Loading P.csv from {data_dir} ===", flush=True)
    P = pd.read_csv(data_dir / "P.csv", index_col="dataset_id")
    R = ap.ranks_from_P(P)
    dataset_ids = list(P.index)
    T_STEPS = P.shape[1]
    n_datasets = len(dataset_ids)
    print(f"P {P.shape}, {n_datasets} datasets, {T_STEPS} algorithms", flush=True)

    variant_X = load_variants(data_dir, dataset_ids)
    for v, df in variant_X.items():
        print(f"  variant {v}: X shape {df.shape}", flush=True)

    treatments = build_treatment_list(args.quick, variant_X)
    print(f"{len(treatments)} treatments: {[t[0] for t in treatments]}", flush=True)

    manifest = None
    group_of = None
    if "loso" in protocols:
        manifest = pd.read_csv(data_dir / "datasets_manifest.csv")
        group_of, groups = build_source_groups(manifest)
        print(f"LOSO source groups: {len(groups)} groups", flush=True)

    for protocol in protocols:
        out_dir = compare_dir / protocol
        out_dir.mkdir(parents=True, exist_ok=True)

        all_results = run_protocol(protocol, treatments, variant_X, P, R, dataset_ids,
                                    group_of, args.n_jobs)
        per_dataset_spearman, per_dataset_auc_loss, mean_loss_curves = assemble_tables(
            treatments, all_results, T_STEPS)

        per_dataset_spearman.to_csv(out_dir / "per_dataset_spearman.csv")
        per_dataset_auc_loss.to_csv(out_dir / "per_dataset_auc_loss.csv")
        mean_loss_curves.to_csv(out_dir / "loss_curves.csv")

        summary_df = build_summary(treatments, per_dataset_spearman, per_dataset_auc_loss,
                                    mean_loss_curves, variant_X)
        summary_df.to_csv(out_dir / "summary.csv", index=False)
        print(f"\n=== [{protocol}] summary ===")
        print(summary_df.round(4).to_string(index=False))

        wilcoxon_df = wilcoxon_vs_base(per_dataset_spearman, per_dataset_auc_loss)
        wilcoxon_df.to_csv(out_dir / "wilcoxon_vs_base.csv", index=False)
        print(f"\n=== [{protocol}] wilcoxon vs base ===")
        print(wilcoxon_df.round(4).to_string(index=False))

        lang_df = language_split(per_dataset_spearman, variant_X["base"], treatments)
        lang_df.to_csv(out_dir / "language_split.csv", index=False)

        fr_sp, fr_auc = friedman_nemenyi_block(per_dataset_spearman, per_dataset_auc_loss, out_dir)
        print(f"\n=== [{protocol}] Friedman (Spearman) statistic={fr_sp['statistic']:.4f} "
              f"p={fr_sp['pvalue']:.6g} CD={fr_sp['cd']:.4f} ===")
        print(f"=== [{protocol}] Friedman (AUC_loss) statistic={fr_auc['statistic']:.4f} "
              f"p={fr_auc['pvalue']:.6g} CD={fr_auc['cd']:.4f} ===")

        plot_delta_spearman(per_dataset_spearman, variant_X["base"], out_dir / "fig_delta_spearman.png")
        plot_loss_curves(mean_loss_curves, T_STEPS, out_dir / "loss_curves.png")

        write_summary_md(out_dir, protocol, summary_df, wilcoxon_df, lang_df, fr_sp, fr_auc, n_datasets)

        print(f"\n=== [{protocol}] done. Files written to {out_dir} ===")
        print(sorted(p.name for p in out_dir.iterdir()))


if __name__ == "__main__":
    main()
