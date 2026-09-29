import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

import approaches as ap
import evaluate as ev

ROOT = Path(__file__).resolve().parent

RANDOM_STATE = 0
LAMS = [0.0, 0.25, 0.5, 0.75, 1.0]


def parse_args():
    parser = argparse.ArgumentParser(
        description="LOO meta-learning experiment pipeline (dimension-agnostic).")
    parser.add_argument("--data-dir", default="data",
                         help="Directory holding X.csv, P.csv, P_folds.csv (default: data)")
    parser.add_argument("--results-dir", default="results",
                         help="Output directory for results, created if missing (default: results)")
    parser.add_argument("--label", default=None,
                         help="Short tag used in progress lines and figure titles to "
                              "distinguish runs (default: derived from --data-dir basename, "
                              "e.g. data_hs -> hs)")
    return parser.parse_args()


def default_label(data_dir):
    name = data_dir.name
    prefix = "data_"
    return name[len(prefix):] if name.startswith(prefix) else name


def guarded_progress(label, n_datasets, every=1, tag=None):
    """Progress callback for evaluate.run_loo; `tag` prefixes the printed lines."""
    prefix = f"{tag}:" if tag else ""
    last_round = n_datasets - 1

    def cb(round_idx, dataset_id):
        if round_idx % 10 == 0 or round_idx == last_round:
            print(f"  [{prefix}{label}] round {round_idx + 1}/{n_datasets} "
                  f"(dataset_id={dataset_id})", flush=True)
    return cb


class RandomOrderBaseline:
    """Random permutation each round; sanity baseline only, excluded from CD diagrams and tests."""
    name = "RandomBaseline"
    needs_X = False

    def __init__(self, rng):
        self.rng = rng

    def fit(self, X_train, P_train, R_train, folds_train=None):
        self.algorithms = list(P_train.columns)
        return self

    def recommend(self, X_query=None):
        order = list(self.rng.permutation(self.algorithms))
        scores = pd.Series({a: i for i, a in enumerate(order)})
        return order, scores


def _df_to_md_table(df):
    """Minimal GitHub-flavoured markdown table (no tabulate dependency)."""
    cols = [str(c) for c in df.columns]
    header = "| " + " | ".join(cols) + " |"
    sep = "| " + " | ".join(["---"] * len(cols)) + " |"
    rows = []
    for _, row in df.iterrows():
        rows.append("| " + " | ".join(str(v) for v in row.values) + " |")
    return "\n".join([header, sep] + rows) + "\n"


def main():
    t_start = time.time()

    args = parse_args()
    data_dir = Path(args.data_dir)
    if not data_dir.is_absolute():
        data_dir = ROOT / data_dir
    results_dir = Path(args.results_dir)
    if not results_dir.is_absolute():
        results_dir = ROOT / results_dir
    results_dir.mkdir(parents=True, exist_ok=True)
    run_tag = args.label if args.label is not None else default_label(data_dir)

    print("=== Loading data ===", flush=True)
    X = pd.read_csv(data_dir / "X.csv", index_col="dataset_id")
    P = pd.read_csv(data_dir / "P.csv", index_col="dataset_id")
    P_folds = pd.read_csv(data_dir / "P_folds.csv")
    R = ap.ranks_from_P(P)
    print(f"X {X.shape}, P {P.shape}, P_folds {P_folds.shape}, R {R.shape}", flush=True)

    dataset_ids = list(P.index)
    algorithms = list(P.columns)
    n_datasets = len(dataset_ids)
    T_STEPS = P.shape[1]  # m = number of algorithms

    print("\n=== Anti-leakage assertion ===", flush=True)
    spot_check_ids = list(dict.fromkeys(
        [dataset_ids[0], dataset_ids[len(dataset_ids) // 2], dataset_ids[-1]]))
    for did in spot_check_ids:
        ev.assert_no_leakage(X, P, R, lambda: ap.AverageRank(), did)
        ev.assert_no_leakage(X, P, R, lambda: ap.RegressorOnP(), did)
    print(f"assert_no_leakage passed for AverageRank and RegressorOnP on "
          f"{len(spot_check_ids)} spot-check dataset(s).", flush=True)

    print("\n=== Precomputing SignificantWins win tensor (ALPHA="
          f"{ap.ALPHA}) over all {n_datasets} datasets, once ===", flush=True)
    t0 = time.time()
    win_tensor = ap.compute_win_tensor(P_folds, dataset_ids, algorithms, alpha=ap.ALPHA)
    print(f"win tensor computed in {time.time() - t0:.2f}s", flush=True)
    ev.assert_no_leakage(X, P, R, lambda: ap.SignificantWins(win_tensor, algorithms), dataset_ids[0])
    print("assert_no_leakage passed for SignificantWins.", flush=True)

    # approach builders
    fixed_builders = {
        "AR": (lambda: ap.AverageRank(), False),
        "MR": (lambda: ap.MedianRank(), False),
        "SW": (lambda: ap.SignificantWins(win_tensor, algorithms), False),
    }
    trained_builders = {
        "RegressorOnP": (lambda: ap.RegressorOnP(random_state=RANDOM_STATE), True),
        "RegressorOnR": (lambda: ap.RegressorOnR(random_state=RANDOM_STATE), True),
    }
    harris_builders = {
        f"Harris(lam={lam})": (
            (lambda lam=lam: ap.Harris(lam=lam, n_estimators=100, random_state=RANDOM_STATE, n_jobs=1)),
            True,
        )
        for lam in LAMS
    }

    all_results = {}  # name -> {"per_dataset_spearman":Series, "loss_curves":DataFrame}

    print("\n=== Running LOO: aggregation approaches (AR, MR, SW) ===", flush=True)
    for name, (builder, needs_X) in fixed_builders.items():
        t0 = time.time()
        res = ev.run_loo(builder, X, P, R, needs_X=needs_X, dataset_ids=dataset_ids,
                          progress_cb=guarded_progress(name, n_datasets, every=10, tag=run_tag))
        all_results[name] = res
        print(f"[{name}] done in {time.time() - t0:.2f}s, mean Spearman = "
              f"{res['per_dataset_spearman'].mean():.4f}", flush=True)

    print("\n=== Running LOO: trained regressor approaches ===", flush=True)
    for name, (builder, needs_X) in trained_builders.items():
        t0 = time.time()
        res = ev.run_loo(builder, X, P, R, needs_X=needs_X, dataset_ids=dataset_ids,
                          progress_cb=guarded_progress(name, n_datasets, every=10, tag=run_tag))
        all_results[name] = res
        print(f"[{name}] done in {time.time() - t0:.2f}s, mean Spearman = "
              f"{res['per_dataset_spearman'].mean():.4f}", flush=True)

    print("\n=== Running LOO: HARRIS lambda sweep ===", flush=True)
    for name, (builder, needs_X) in harris_builders.items():
        t0 = time.time()
        res = ev.run_loo(builder, X, P, R, needs_X=needs_X, dataset_ids=dataset_ids,
                          progress_cb=guarded_progress(name, n_datasets, every=1, tag=run_tag))
        all_results[name] = res
        print(f"[{name}] done in {time.time() - t0:.2f}s, mean Spearman = "
              f"{res['per_dataset_spearman'].mean():.4f}", flush=True)

    print("\n=== Running LOO: RandomOrder sanity baseline (not in CD diagram) ===", flush=True)
    rng = np.random.default_rng(RANDOM_STATE)
    res = ev.run_loo(lambda: RandomOrderBaseline(rng), X, P, R, needs_X=False,
                      dataset_ids=dataset_ids,
                      progress_cb=guarded_progress("RandomBaseline", n_datasets, every=10, tag=run_tag))
    all_results["RandomBaseline"] = res
    print(f"[RandomBaseline] mean Spearman = {res['per_dataset_spearman'].mean():.4f} "
          f"(expect close to 0)", flush=True)

    # select best HARRIS lam by mean Spearman
    harris_names = list(harris_builders.keys())
    harris_mean_spearman = {n: all_results[n]["per_dataset_spearman"].mean() for n in harris_names}
    best_harris_name = max(harris_mean_spearman, key=harris_mean_spearman.get)
    best_lam = float(best_harris_name.split("lam=")[1].rstrip(")"))
    print(f"\n=== Selected HARRIS lam = {best_lam} (mean Spearman = "
          f"{harris_mean_spearman[best_harris_name]:.4f}) for the six-approach comparison ===",
          flush=True)
    all_results["Harris"] = all_results[best_harris_name]

    # assemble tables
    approach_order_full = ["AR", "MR", "SW", "RegressorOnP", "RegressorOnR"] + harris_names + \
        ["Harris", "RandomBaseline"]
    six_approaches = ["AR", "MR", "SW", "RegressorOnP", "RegressorOnR", "Harris"]

    per_dataset_spearman = pd.DataFrame(
        {name: all_results[name]["per_dataset_spearman"] for name in approach_order_full})
    per_dataset_spearman.index.name = "dataset_id"

    per_dataset_auc_loss = pd.DataFrame(
        {name: ev.auc_loss_from_curve_df(all_results[name]["loss_curves"])
         for name in approach_order_full})
    per_dataset_auc_loss.index.name = "dataset_id"

    spearman_summary = pd.DataFrame({
        "approach": approach_order_full,
        "mean_spearman": [per_dataset_spearman[n].mean() for n in approach_order_full],
        "std_spearman": [per_dataset_spearman[n].std() for n in approach_order_full],
    })

    mean_loss_curves = pd.DataFrame(
        {name: all_results[name]["loss_curves"].mean(axis=0).values for name in approach_order_full},
        index=[f"t{t}" for t in range(1, T_STEPS + 1)]).T
    mean_loss_curves.index.name = "approach"

    auc_loss_summary = pd.DataFrame({
        "approach": approach_order_full,
        "mean_auc_loss": [per_dataset_auc_loss[n].mean() for n in approach_order_full],
        "std_auc_loss": [per_dataset_auc_loss[n].std() for n in approach_order_full],
    })

    harris_lambda_table = pd.DataFrame({
        "lam": LAMS,
        "mean_spearman": [per_dataset_spearman[n].mean() for n in harris_names],
        "std_spearman": [per_dataset_spearman[n].std() for n in harris_names],
        "mean_auc_loss": [per_dataset_auc_loss[n].mean() for n in harris_names],
        "std_auc_loss": [per_dataset_auc_loss[n].std() for n in harris_names],
    })

    print("\n=== Friedman + Nemenyi (Spearman) over the six approaches ===", flush=True)
    fr_spearman = ev.friedman_nemenyi(per_dataset_spearman[six_approaches], higher_is_better=True)
    print(f"statistic={fr_spearman['statistic']:.4f} p={fr_spearman['pvalue']:.6g} "
          f"CD={fr_spearman['cd']:.4f}", flush=True)
    print("mean ranks:", fr_spearman["mean_ranks"].round(3).to_dict(), flush=True)
    print("significant pairs (>=CD):", fr_spearman["significant_pairs"], flush=True)

    print("\n=== Friedman + Nemenyi (AUC_loss) over the six approaches ===", flush=True)
    fr_aucloss = ev.friedman_nemenyi(per_dataset_auc_loss[six_approaches], higher_is_better=False)
    print(f"statistic={fr_aucloss['statistic']:.4f} p={fr_aucloss['pvalue']:.6g} "
          f"CD={fr_aucloss['cd']:.4f}", flush=True)
    print("mean ranks:", fr_aucloss["mean_ranks"].round(3).to_dict(), flush=True)
    print("significant pairs (>=CD):", fr_aucloss["significant_pairs"], flush=True)

    print(f"\n=== Writing {results_dir}/ ===", flush=True)
    spearman_summary.to_csv(results_dir / "spearman.csv", index=False)
    mean_loss_curves.to_csv(results_dir / "loss_curves.csv")
    auc_loss_summary.to_csv(results_dir / "auc_loss.csv", index=False)
    per_dataset_spearman.to_csv(results_dir / "per_dataset_spearman.csv")
    per_dataset_auc_loss.to_csv(results_dir / "per_dataset_auc_loss.csv")
    harris_lambda_table.to_csv(results_dir / "harris_lambda.csv", index=False)

    # plots
    ev.plot_cd_diagram(fr_spearman["mean_ranks"], fr_spearman["cd"],
                        f"CD diagram -- mean Spearman rank (CD={fr_spearman['cd']:.3f}, "
                        f"HARRIS lam={best_lam}) [{run_tag}]",
                        str(results_dir / "cd_diagram_spearman.png"))
    ev.plot_cd_diagram(fr_aucloss["mean_ranks"], fr_aucloss["cd"],
                        f"CD diagram -- mean AUC_loss rank (CD={fr_aucloss['cd']:.3f}, "
                        f"HARRIS lam={best_lam}) [{run_tag}]",
                        str(results_dir / "cd_diagram_aucloss.png"))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 5.5))
    ts = np.arange(1, T_STEPS + 1)
    for name in six_approaches:
        ax.plot(ts, mean_loss_curves.loc[name].values, marker="o", label=name)
    ax.plot(ts, mean_loss_curves.loc["RandomBaseline"].values, linestyle="--", color="gray",
            label="RandomBaseline (reference only)")
    ax.set_xlabel("number of tests t")
    ax.set_ylabel("mean loss (accuracy points)")
    ax.set_title(f"Mean loss curve by number of tests (LOO over {n_datasets} datasets) [{run_tag}]")
    ax.set_xticks(ts)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(results_dir / "loss_curves.png", dpi=150)
    plt.close(fig)

    # friedman_nemenyi.txt
    lines = []
    lines.append("Friedman + Nemenyi post-hoc analysis")
    lines.append("=" * 60)
    lines.append(f"k = 6 approaches, N = {n_datasets} datasets, alpha = 0.05")
    lines.append(f"HARRIS enters as ONE approach: selected lam = {best_lam} "
                 f"(best mean Spearman = {harris_mean_spearman[best_harris_name]:.4f} among "
                 f"{ {n: round(v, 4) for n, v in harris_mean_spearman.items()} })")
    lines.append("")
    lines.append("--- Metric: per-dataset Spearman correlation (higher = better) ---")
    lines.append(f"Friedman statistic = {fr_spearman['statistic']:.4f}, "
                 f"p-value = {fr_spearman['pvalue']:.6g}")
    lines.append(f"Nemenyi q_alpha(k=6, alpha=0.05) = {fr_spearman['q_alpha']}")
    lines.append(f"CD = q_alpha * sqrt(k(k+1)/(6N)) = {fr_spearman['cd']:.4f}")
    lines.append("Mean ranks (rank 1 = best):")
    for name, r in fr_spearman["mean_ranks"].sort_values().items():
        lines.append(f"  {name:16s} {r:.3f}")
    lines.append("Pairs with |mean rank diff| >= CD (significantly different):")
    if fr_spearman["significant_pairs"]:
        for a, b, d in fr_spearman["significant_pairs"]:
            lines.append(f"  {a} vs {b}: diff = {d:.3f}")
    else:
        lines.append("  (none)")
    lines.append("Pairs with |mean rank diff| < CD (NOT significantly different):")
    for a, b, d in fr_spearman["nonsignificant_pairs"]:
        lines.append(f"  {a} vs {b}: diff = {d:.3f}")

    lines.append("")
    lines.append(f"--- Metric: per-dataset AUC_loss (mean loss over t=1..{T_STEPS}, lower = better) ---")
    lines.append(f"Friedman statistic = {fr_aucloss['statistic']:.4f}, "
                 f"p-value = {fr_aucloss['pvalue']:.6g}")
    lines.append(f"Nemenyi q_alpha(k=6, alpha=0.05) = {fr_aucloss['q_alpha']}")
    lines.append(f"CD = q_alpha * sqrt(k(k+1)/(6N)) = {fr_aucloss['cd']:.4f}")
    lines.append("Mean ranks (rank 1 = best, i.e. lowest AUC_loss):")
    for name, r in fr_aucloss["mean_ranks"].sort_values().items():
        lines.append(f"  {name:16s} {r:.3f}")
    lines.append("Pairs with |mean rank diff| >= CD (significantly different):")
    if fr_aucloss["significant_pairs"]:
        for a, b, d in fr_aucloss["significant_pairs"]:
            lines.append(f"  {a} vs {b}: diff = {d:.3f}")
    else:
        lines.append("  (none)")
    lines.append("Pairs with |mean rank diff| < CD (NOT significantly different):")
    for a, b, d in fr_aucloss["nonsignificant_pairs"]:
        lines.append(f"  {a} vs {b}: diff = {d:.3f}")

    (results_dir / "friedman_nemenyi.txt").write_text("\n".join(lines) + "\n")

    print("\n=== Sanity checks ===", flush=True)
    rand_mean = per_dataset_spearman["RandomBaseline"].mean()
    rand_auc = per_dataset_auc_loss["RandomBaseline"].mean()
    best_real_auc = auc_loss_summary.set_index("approach").loc[six_approaches, "mean_auc_loss"].min()
    print(f"RandomBaseline mean Spearman = {rand_mean:.4f} (expect near 0)", flush=True)
    print(f"RandomBaseline mean AUC_loss = {rand_auc:.4f} vs best real approach AUC_loss = "
          f"{best_real_auc:.4f} (expect random much worse / higher)", flush=True)
    lam0_ok = np.isfinite(per_dataset_spearman["Harris(lam=0.0)"]).all()
    lam1_ok = np.isfinite(per_dataset_spearman["Harris(lam=1.0)"]).all()
    print(f"lam=0 finite for all {n_datasets} datasets: {lam0_ok}; "
          f"lam=1 finite for all {n_datasets} datasets: {lam1_ok}",
          flush=True)

    # summary.md
    runtime = time.time() - t_start
    md = []
    md.append("# Experimental results summary\n")
    md.append(f"Total runtime: {runtime:.1f}s.\n")
    md.append(f"HARRIS lam swept over {LAMS}; selected lam = **{best_lam}** for the six-approach "
              "comparison (best mean Spearman).\n")
    md.append("## Mean Spearman correlation and AUC_loss (all approaches)\n")
    tbl = spearman_summary.merge(auc_loss_summary, on="approach")
    md.append(_df_to_md_table(tbl.round(4)))
    md.append("\n\n## HARRIS lambda sweep\n")
    md.append(_df_to_md_table(harris_lambda_table.round(4)))
    md.append(f"\n\n## Loss curve (mean loss at t=1..{T_STEPS}) -- six approaches + RandomBaseline\n")
    md.append(_df_to_md_table(
        mean_loss_curves.loc[six_approaches + ["RandomBaseline"]].round(4).reset_index()))
    md.append("\n\n## Friedman + Nemenyi -- Spearman (six approaches)\n")
    md.append(f"statistic = {fr_spearman['statistic']:.4f}, p = {fr_spearman['pvalue']:.6g}, "
              f"CD = {fr_spearman['cd']:.4f}\n")
    md.append("Mean ranks: " + ", ".join(
        f"{n}={r:.3f}" for n, r in fr_spearman["mean_ranks"].sort_values().items()) + "\n")
    md.append("Significant pairs: " + (
        ", ".join(f"{a} vs {b} (d={d:.3f})" for a, b, d in fr_spearman["significant_pairs"])
        or "none") + "\n")
    md.append("\n## Friedman + Nemenyi -- AUC_loss (six approaches)\n")
    md.append(f"statistic = {fr_aucloss['statistic']:.4f}, p = {fr_aucloss['pvalue']:.6g}, "
              f"CD = {fr_aucloss['cd']:.4f}\n")
    md.append("Mean ranks: " + ", ".join(
        f"{n}={r:.3f}" for n, r in fr_aucloss["mean_ranks"].sort_values().items()) + "\n")
    md.append("Significant pairs: " + (
        ", ".join(f"{a} vs {b} (d={d:.3f})" for a, b, d in fr_aucloss["significant_pairs"])
        or "none") + "\n")
    md.append("\n## Sanity checks\n")
    md.append(f"- RandomBaseline mean Spearman = {rand_mean:.4f} (expect near 0)\n")
    md.append(f"- RandomBaseline mean AUC_loss = {rand_auc:.4f} vs best six-approach AUC_loss = "
              f"{best_real_auc:.4f}\n")
    md.append(f"- lam=0 finite: {lam0_ok}; lam=1 finite: {lam1_ok}\n")
    md.append(f"- ALPHA (SignificantWins) = {ap.ALPHA}\n")
    (results_dir / "summary.md").write_text("".join(md))

    print(f"\n=== Done in {runtime:.1f}s ===", flush=True)
    print(f"Files written to {results_dir}/:", sorted(p.name for p in results_dir.iterdir()), flush=True)


if __name__ == "__main__":
    main()
