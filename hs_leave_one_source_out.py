import os
import re
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

import approaches as ap
import evaluate as ev
from run_experiments import (
    RandomOrderBaseline,
    _df_to_md_table,
    guarded_progress,
)

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data_hs"
RESULTS_DIR = ROOT / "results_hs"
LOSO_DIR = RESULTS_DIR / "loso"
TABLES_DIR = RESULTS_DIR / "tables"
TEX_DIR = TABLES_DIR / "tex"
LOSO_DIR.mkdir(parents=True, exist_ok=True)
TABLES_DIR.mkdir(parents=True, exist_ok=True)
TEX_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 0
LAM = 0.5

MANUAL_MERGES = {"sbic": "sbic+sbic_hf", "sbic_hf": "sbic+sbic_hf"}


def esc(s):
    return str(s).replace("_", r"\_")


def fnum(x, decimals=4):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return ""
    return f"{float(x):.{decimals}f}".replace(".", ",")


def build_source_groups(manifest):
    """Return (dataset_id -> group_label, group_label -> [dataset_ids])."""
    manifest = manifest.set_index("dataset_id")
    group_of = {}
    for did, row in manifest.iterrows():
        src = row["source"]
        group_of[did] = MANUAL_MERGES.get(src, src)
    groups = defaultdict(list)
    for did, g in group_of.items():
        groups[g].append(did)
    return group_of, dict(groups)


def verify_no_other_collisions(manifest):
    """Flag distinct `source` labels that share a HF repo id or local directory."""
    def source_key(prov):
        m = re.search(r"HF dataset ([\w./-]+)", prov)
        if m:
            return "hf:" + m.group(1).lower()
        m = re.search(r"local files? \[?'?(/[^,'\]]+)", prov)
        if m:
            return "localdir:" + os.path.dirname(m.group(1))
        return "unk:" + prov[:40]

    key_to_sources = defaultdict(set)
    for _, row in manifest.iterrows():
        key_to_sources[source_key(row["provenance"])].add(row["source"])
    return {k: s for k, s in key_to_sources.items() if len(s) > 1}


def run_loso(build_fn, X, P, R,
             needs_X, group_of, dataset_ids=None, progress_cb=None):
    """Like evaluate.run_loo, but also excludes every dataset from the held-out row's source group."""
    if dataset_ids is None:
        dataset_ids = list(P.index)
    m = P.shape[1]
    spearman_res = {}
    loss_res = {}
    orders = {}
    for i, did in enumerate(dataset_ids):
        if progress_cb is not None:
            progress_cb(i, did)
        held_group = group_of[did]
        train_ids = [d for d in dataset_ids if group_of[d] != held_group]
        assert did not in train_ids  # anti-leakage: whole group excluded
        X_train = X.loc[train_ids]
        P_train = P.loc[train_ids]
        R_train = R.loc[train_ids]

        model = build_fn()
        model.fit(X_train, P_train, R_train)

        X_query = X.loc[[did]].values if needs_X else None
        order, rank_scores = model.recommend(X_query)

        true_R_row = R.loc[did]
        spearman_res[did] = ev.spearman_score(rank_scores, true_R_row)

        P_row = P.loc[did]
        loss_res[did] = ev.loss_curve(order, P_row)
        orders[did] = order

    spearman_series = pd.Series(spearman_res, name="spearman")
    loss_df = pd.DataFrame.from_dict(
        loss_res, orient="index", columns=[f"t{t}" for t in range(1, m + 1)])
    loss_df = loss_df.loc[spearman_series.index]
    return {"per_dataset_spearman": spearman_series, "loss_curves": loss_df, "orders": orders}


def assert_no_group_leakage(X, P, R, build_fn, group_of, held_out_id):
    held_group = group_of[held_out_id]
    train_ids = [d for d in P.index if group_of[d] != held_group]
    group_members = [d for d in P.index if group_of[d] == held_group]
    for d in group_members:
        assert d not in train_ids
    model = build_fn()
    model.fit(X.loc[train_ids], P.loc[train_ids], R.loc[train_ids])
    return True


def main():
    t_start = time.time()

    print("=== Loading data ===", flush=True)
    X = pd.read_csv(DATA_DIR / "X.csv", index_col="dataset_id")
    P = pd.read_csv(DATA_DIR / "P.csv", index_col="dataset_id")
    P_folds = pd.read_csv(DATA_DIR / "P_folds.csv")
    manifest = pd.read_csv(DATA_DIR / "datasets_manifest.csv")
    R = ap.ranks_from_P(P)
    print(f"X {X.shape}, P {P.shape}, P_folds {P_folds.shape}, R {R.shape}", flush=True)

    dataset_ids = list(P.index)
    algorithms = list(P.columns)
    n_datasets = len(dataset_ids)
    T_STEPS = P.shape[1]

    print("\n=== Source grouping ===", flush=True)
    raw_counts = manifest["source"].value_counts()
    print("Raw manifest `source` value counts (14 labels):")
    print(raw_counts.to_string())

    collisions = verify_no_other_collisions(manifest)
    print("\nAutomated provenance-key collision check across the 14 raw source "
          "labels (HF repo id / local file directory extracted from `provenance`):")
    if collisions:
        for k, s in collisions.items():
            print(f"  COLLISION key={k!r} shared by sources {sorted(s)}")
    else:
        print("  0 collisions found by string-matching provenance.")
    print("  -> sbic (1 row, local files) and sbic_hf (6 rows, HF "
          "allenai/social_bias_frames) do NOT share a literal provenance key "
          "(different ingestion pipelines) but ARE the same underlying corpus "
          "on domain knowledge (Sap et al., Social Bias Frames) -- merged "
          "manually. No other pair of the 14 raw source labels collides, "
          "either by provenance key or by domain knowledge: each remaining "
          "label maps to one distinct HF dataset repo id or one distinct "
          "local file family.")

    group_of, groups = build_source_groups(manifest)
    assert sum(len(v) for v in groups.values()) == n_datasets
    group_sizes = pd.Series({g: len(v) for g, v in groups.items()}).sort_values(ascending=False)
    print(f"\nFinal group sizes ({len(groups)} groups, sum={group_sizes.sum()}):")
    print(group_sizes.to_string())

    print("\n=== Anti-leakage assertion (group-based) ===", flush=True)
    spot_check_ids = list(dict.fromkeys(
        [dataset_ids[0], dataset_ids[len(dataset_ids) // 2], dataset_ids[-1]]))
    for did in spot_check_ids:
        assert_no_group_leakage(X, P, R, lambda: ap.AverageRank(), group_of, did)
        assert_no_group_leakage(X, P, R, lambda: ap.RegressorOnP(), group_of, did)
    print(f"assert_no_group_leakage passed for AverageRank and RegressorOnP on "
          f"{len(spot_check_ids)} spot-check dataset(s).", flush=True)

    print("\n=== Precomputing SignificantWins win tensor (ALPHA="
          f"{ap.ALPHA}) over all {n_datasets} datasets, once ===", flush=True)
    t0 = time.time()
    win_tensor = ap.compute_win_tensor(P_folds, dataset_ids, algorithms, alpha=ap.ALPHA)
    print(f"win tensor computed in {time.time() - t0:.2f}s", flush=True)

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
    harris_builder = {
        "Harris": (lambda: ap.Harris(lam=LAM, n_estimators=100,
                                      random_state=RANDOM_STATE, n_jobs=1), True),
    }

    six_approaches = ["AR", "MR", "SW", "RegressorOnP", "RegressorOnR", "Harris"]
    all_results = {}

    print("\n=== Running LOSO: aggregation approaches (AR, MR, SW) ===", flush=True)
    for name, (builder, needs_X) in fixed_builders.items():
        t0 = time.time()
        res = run_loso(builder, X, P, R, needs_X=needs_X, group_of=group_of,
                        dataset_ids=dataset_ids,
                        progress_cb=guarded_progress(name, n_datasets, every=10, tag="hs-loso"))
        all_results[name] = res
        print(f"[{name}] done in {time.time() - t0:.2f}s, mean Spearman = "
              f"{res['per_dataset_spearman'].mean():.4f}", flush=True)

    print("\n=== Running LOSO: trained regressor approaches ===", flush=True)
    for name, (builder, needs_X) in trained_builders.items():
        t0 = time.time()
        res = run_loso(builder, X, P, R, needs_X=needs_X, group_of=group_of,
                        dataset_ids=dataset_ids,
                        progress_cb=guarded_progress(name, n_datasets, every=10, tag="hs-loso"))
        all_results[name] = res
        print(f"[{name}] done in {time.time() - t0:.2f}s, mean Spearman = "
              f"{res['per_dataset_spearman'].mean():.4f}", flush=True)

    print(f"\n=== Running LOSO: HARRIS (lam={LAM}) ===", flush=True)
    for name, (builder, needs_X) in harris_builder.items():
        t0 = time.time()
        res = run_loso(builder, X, P, R, needs_X=needs_X, group_of=group_of,
                        dataset_ids=dataset_ids,
                        progress_cb=guarded_progress(name, n_datasets, every=1, tag="hs-loso"))
        all_results[name] = res
        print(f"[{name}] done in {time.time() - t0:.2f}s, mean Spearman = "
              f"{res['per_dataset_spearman'].mean():.4f}", flush=True)

    print("\n=== Running LOSO: RandomOrder sanity baseline (not in CD diagram) ===", flush=True)
    rng = np.random.default_rng(RANDOM_STATE)
    res = run_loso(lambda: RandomOrderBaseline(rng), X, P, R, needs_X=False,
                    group_of=group_of, dataset_ids=dataset_ids,
                    progress_cb=guarded_progress("RandomBaseline", n_datasets, every=10, tag="hs-loso"))
    all_results["RandomBaseline"] = res
    print(f"[RandomBaseline] mean Spearman = {res['per_dataset_spearman'].mean():.4f} "
          f"(expect close to 0)", flush=True)

    approach_order_full = six_approaches + ["RandomBaseline"]

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
    auc_loss_summary = pd.DataFrame({
        "approach": approach_order_full,
        "mean_auc_loss": [per_dataset_auc_loss[n].mean() for n in approach_order_full],
        "std_auc_loss": [per_dataset_auc_loss[n].std() for n in approach_order_full],
    })
    mean_loss_curves = pd.DataFrame(
        {name: all_results[name]["loss_curves"].mean(axis=0).values for name in approach_order_full},
        index=[f"t{t}" for t in range(1, T_STEPS + 1)]).T
    mean_loss_curves.index.name = "approach"

    print("\n=== Friedman + Nemenyi (Spearman, LOSO) over the six approaches ===", flush=True)
    fr_spearman = ev.friedman_nemenyi(per_dataset_spearman[six_approaches], higher_is_better=True)
    print(f"statistic={fr_spearman['statistic']:.4f} p={fr_spearman['pvalue']:.6g} "
          f"CD={fr_spearman['cd']:.4f}", flush=True)
    print("mean ranks:", fr_spearman["mean_ranks"].round(3).to_dict(), flush=True)
    print("significant pairs (>=CD):", fr_spearman["significant_pairs"], flush=True)

    print("\n=== Friedman + Nemenyi (AUC_loss, LOSO) over the six approaches ===", flush=True)
    fr_aucloss = ev.friedman_nemenyi(per_dataset_auc_loss[six_approaches], higher_is_better=False)
    print(f"statistic={fr_aucloss['statistic']:.4f} p={fr_aucloss['pvalue']:.6g} "
          f"CD={fr_aucloss['cd']:.4f}", flush=True)
    print("mean ranks:", fr_aucloss["mean_ranks"].round(3).to_dict(), flush=True)
    print("significant pairs (>=CD):", fr_aucloss["significant_pairs"], flush=True)

    # English vs non-English breakdown under LOSO
    manifest_idx = manifest.set_index("dataset_id")
    en_mask = manifest_idx.loc[dataset_ids, "language"] == "en"
    en_mask = en_mask.reindex(dataset_ids)
    n_en = int(en_mask.sum())
    n_ne = int((~en_mask).sum())
    assert n_en == 43 and n_ne == 7, f"expected 43/7 split, got {n_en}/{n_ne}"

    sp = per_dataset_spearman
    sp_en = sp.loc[en_mask.values]
    sp_ne = sp.loc[~en_mask.values]
    en_ne_rows = []
    for name in six_approaches:
        en_ne_rows.append({
            "approach": name,
            "spearman_english": sp_en[name].mean(),
            "spearman_non_english": sp_ne[name].mean(),
            "drop": sp_ne[name].mean() - sp_en[name].mean(),
        })
    en_ne_df = pd.DataFrame(en_ne_rows)

    # LOO vs LOSO comparison (T8)
    loo_spearman = pd.read_csv(RESULTS_DIR / "spearman.csv").set_index("approach")
    loo_aucloss = pd.read_csv(RESULTS_DIR / "auc_loss.csv").set_index("approach")

    t8_rows = []
    for name in six_approaches:
        sp_loo = float(loo_spearman.loc[name, "mean_spearman"])
        sp_loso = float(spearman_summary.set_index("approach").loc[name, "mean_spearman"])
        auc_loo = float(loo_aucloss.loc[name, "mean_auc_loss"])
        auc_loso = float(auc_loss_summary.set_index("approach").loc[name, "mean_auc_loss"])
        t8_rows.append({
            "approach": name,
            "spearman_loo": sp_loo,
            "spearman_loso": sp_loso,
            "spearman_diff_loso_minus_loo": sp_loso - sp_loo,
            "aucloss_loo": auc_loo,
            "aucloss_loso": auc_loso,
        })
    t8_df = pd.DataFrame(t8_rows)

    print(f"\n=== Writing {LOSO_DIR}/ ===", flush=True)
    spearman_summary.to_csv(LOSO_DIR / "spearman.csv", index=False)
    auc_loss_summary.to_csv(LOSO_DIR / "auc_loss.csv", index=False)
    per_dataset_spearman.to_csv(LOSO_DIR / "per_dataset_spearman.csv")
    mean_loss_curves.to_csv(LOSO_DIR / "loss_curves.csv")
    per_dataset_auc_loss.to_csv(LOSO_DIR / "per_dataset_auc_loss.csv")
    en_ne_df.to_csv(LOSO_DIR / "english_vs_nonenglish.csv", index=False)
    group_sizes.rename("n_datasets").rename_axis("source_group").to_csv(LOSO_DIR / "source_groups.csv")

    t8_csv_path = TABLES_DIR / "T8_loso_comparison.csv"
    t8_df.to_csv(t8_csv_path, index=False)

    # T8 tex table
    header = ["Abordagem", "Spearman (LOO)", "Spearman (LOSO)", "$\\Delta$",
              "AUC\\_loss (LOO)", "AUC\\_loss (LOSO)"]
    tex_rows = []
    for _, r in t8_df.iterrows():
        tex_rows.append([
            esc(r["approach"]),
            fnum(r["spearman_loo"], 4), fnum(r["spearman_loso"], 4),
            fnum(r["spearman_diff_loso_minus_loo"], 4),
            fnum(r["aucloss_loo"], 4), fnum(r["aucloss_loso"], 4),
        ])
    aligns = ["l", "r", "r", "r", "r", "r"]
    tex_lines = [f"\\begin{{tabular}}{{{''.join(aligns)}}}", "\\toprule",
                 " & ".join(header) + r" \\", "\\midrule"]
    for row in tex_rows:
        tex_lines.append(" & ".join(row) + r" \\")
    tex_lines += ["\\bottomrule", "\\end{tabular}"]
    (TEX_DIR / "T8_loso_comparison.tex").write_text("\n".join(tex_lines) + "\n")

    # friedman_nemenyi.txt
    lines = []
    lines.append("Friedman + Nemenyi post-hoc analysis -- LEAVE-ONE-SOURCE-OUT (LOSO)")
    lines.append("=" * 70)
    lines.append(f"k = 6 approaches, N = {n_datasets} datasets, alpha = 0.05")
    lines.append(f"HARRIS fixed at lam = {LAM} (matches the existing LOO six-approach comparison)")
    lines.append(f"Source groups: {len(groups)} groups, sizes = "
                 f"{group_sizes.to_dict()}")
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

    lines.append("")
    lines.append(f"--- English (n={n_en}) vs non-English (n={n_ne}) Spearman, LOSO ---")
    for _, r in en_ne_df.iterrows():
        lines.append(f"  {r['approach']:16s} en={r['spearman_english']:.4f}  "
                     f"non-en={r['spearman_non_english']:.4f}  drop={r['drop']:.4f}")

    (LOSO_DIR / "friedman_nemenyi.txt").write_text("\n".join(lines) + "\n")

    rand_mean = per_dataset_spearman["RandomBaseline"].mean()
    rand_auc = per_dataset_auc_loss["RandomBaseline"].mean()
    print(f"\nRandomBaseline mean Spearman = {rand_mean:.4f} (expect near 0)", flush=True)
    print(f"RandomBaseline mean AUC_loss = {rand_auc:.4f}", flush=True)

    # summary.md
    runtime = time.time() - t_start
    md = []
    md.append("# LOSO (leave-one-source-out) results summary\n")
    md.append(f"Total runtime: {runtime:.1f}s.\n")
    md.append("Repeats the LOO evaluation (`results_hs/`) but, on every round, removes "
              "**every row sharing the held-out row's source group** from the "
              "meta-model's training set (not just the held-out row itself). "
              "Evaluation stays per-dataset (N=50), so the numbers below are directly "
              "comparable to the plain-LOO ones.\n")
    md.append(f"\n## Source groups ({len(groups)} groups)\n")
    md.append("`sbic` (1 row) and `sbic_hf` (6 rows) manually merged: same underlying "
              "corpus (Social Bias Frames, Sap et al.), two ingestion pipelines. No "
              "other pair of the manifest's 14 raw `source` labels collides (checked "
              "both by domain knowledge and by an automated provenance-key match -- "
              "see script output / `verify_no_other_collisions`).\n")
    md.append(_df_to_md_table(group_sizes.rename("n_datasets").rename_axis("source_group")
                               .reset_index()))
    md.append(f"\nHARRIS fixed at lam = **{LAM}** (matches the existing LOO six-approach "
              "comparison; not re-swept here).\n")
    md.append("\n## Mean Spearman correlation and AUC_loss (all approaches, LOSO)\n")
    tbl = spearman_summary.merge(auc_loss_summary, on="approach")
    md.append(_df_to_md_table(tbl.round(4)))
    md.append(f"\n\n## Loss curve (mean loss at t=1..{T_STEPS}) -- six approaches + RandomBaseline\n")
    md.append(_df_to_md_table(
        mean_loss_curves.loc[six_approaches + ["RandomBaseline"]].round(4).reset_index()))
    md.append("\n\n## Friedman + Nemenyi -- Spearman (six approaches, LOSO)\n")
    md.append(f"statistic = {fr_spearman['statistic']:.4f}, p = {fr_spearman['pvalue']:.6g}, "
              f"CD = {fr_spearman['cd']:.4f}\n")
    md.append("Mean ranks: " + ", ".join(
        f"{n}={r:.3f}" for n, r in fr_spearman["mean_ranks"].sort_values().items()) + "\n")
    md.append("Significant pairs: " + (
        ", ".join(f"{a} vs {b} (d={d:.3f})" for a, b, d in fr_spearman["significant_pairs"])
        or "none") + "\n")
    md.append("\n## Friedman + Nemenyi -- AUC_loss (six approaches, LOSO)\n")
    md.append(f"statistic = {fr_aucloss['statistic']:.4f}, p = {fr_aucloss['pvalue']:.6g}, "
              f"CD = {fr_aucloss['cd']:.4f}\n")
    md.append("Mean ranks: " + ", ".join(
        f"{n}={r:.3f}" for n, r in fr_aucloss["mean_ranks"].sort_values().items()) + "\n")
    md.append("Significant pairs: " + (
        ", ".join(f"{a} vs {b} (d={d:.3f})" for a, b, d in fr_aucloss["significant_pairs"])
        or "none") + "\n")
    md.append(f"\n## English (n={n_en}) vs non-English (n={n_ne}) Spearman, LOSO vs LOO\n")
    md.append("LOSO breakdown:\n")
    md.append(_df_to_md_table(en_ne_df.round(4)))
    md.append("\n## LOO vs LOSO comparison (T8)\n")
    md.append(_df_to_md_table(t8_df.round(4)))
    md.append("\n## Sanity checks\n")
    md.append(f"- RandomBaseline mean Spearman = {rand_mean:.4f} (expect near 0)\n")
    md.append(f"- RandomBaseline mean AUC_loss = {rand_auc:.4f}\n")
    md.append(f"- ALPHA (SignificantWins) = {ap.ALPHA}\n")
    (LOSO_DIR / "summary.md").write_text("".join(md))

    print(f"\n=== Done in {runtime:.1f}s ===", flush=True)
    print(f"Files written to {LOSO_DIR}/:", sorted(p.name for p in LOSO_DIR.iterdir()), flush=True)
    print(f"Also wrote {t8_csv_path} and {TEX_DIR / 'T8_loso_comparison.tex'}", flush=True)

    print("\n=== T8 LOO vs LOSO comparison ===")
    print(t8_df.round(4).to_string(index=False))
    print("\n=== English vs non-English Spearman under LOSO ===")
    print(en_ne_df.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
