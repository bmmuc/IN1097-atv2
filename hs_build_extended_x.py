#!/usr/bin/env python3

import argparse
import os
import time

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))

import hs_meta_features as mf  # noqa: E402
import hs_domain_meta_features as dmf  # noqa: E402

RANDOM_STATE = 42

# column order per new group (as returned by hs_domain_meta_features.compute_<group>_features)
NEW_COLUMNS_BY_GROUP = {
    "tok": ["tok_fertility_hatebert", "tok_continued_hatebert", "tok_unk_hatebert",
            "tok_fertility_mbert", "tok_continued_mbert", "tok_unk_mbert", "tok_fertility_gap"],
    "hl": ["hl_doc_rate", "hl_token_rate", "hl_class_gap", "hl_category_entropy"],
    "ch": ["ch_js_between_classes", "ch_js_class_to_corpus", "ch_vocab_jaccard", "ch_zipf_slope"],
    "lm": ["lm_stump_bow_f1", "lm_tree3_bow_f1", "lm_nc_tfidf_f1", "lm_nc_minilm_f1",
           "lm_nc_hatebert_f1", "lm_nc_gap_minilm_minus_tfidf"],
    "cx": ["cx_f1", "cx_n1", "cx_n2", "cx_lsc", "cx_t2", "cx_t3", "cx_t4",
           "cx_density", "cx_clscoef", "cx_hubs"],
    "ih": ["ih_kdn_minilm", "ih_kdn_hatebert", "ih_kdn_mbert", "ih_kdn_tfidf",
           "ih_hard_frac_minilm", "ih_kdn_gap_tfidf_minus_minilm"],
    "mt": ["mt_safe", "mt_borderline", "mt_rare", "mt_outlier", "mt_safe_tfidf", "mt_outlier_tfidf"],
    "lc": ["lc_slope_minilm", "lc_slope_tfidf", "lc_auc_gap_minilm_minus_tfidf"],
}
NEW_GROUP_ORDER = ["tok", "hl", "ch", "lm", "cx", "ih", "mt", "lc"]
NEW_COLUMNS = [c for g in NEW_GROUP_ORDER for c in NEW_COLUMNS_BY_GROUP[g]]

NEW_GROUP_INFO = {  # group -> (family label in Portuguese, reference)
    "tok": ("tokenização (domínio)", "Rust et al. 2021 ACL 'How good is your tokenizer?'"),
    "hl": ("léxico de ódio (domínio)", "Bassignana, Basile & Patti 2018 HurtLex; Wiegand et al. 2019 NAACL"),
    "ch": ("dureza de corpus (domínio)", "Pinto & Rosso 2007 TSD; Madrid, Escalante & Morales 2019 CIARP"),
    "lm": ("landmarker", "Pfahringer, Bensusan & Giraud-Carrier 2000 ICML; Furnkranz & Petrak 2001"),
    "cx": ("complexidade", "Ho & Basu 2002 TPAMI; Lorena et al. 2019 ACM CSUR; Garcia, de Carvalho & Lorena 2015"),
    "ih": ("dureza de instância", "Smith, Martinez & Giraud-Carrier 2014 Machine Learning"),
    "mt": ("tipologia de minoria", "Napierala & Stefanowski 2016 JIIS"),
    "lc": ("curva de aprendizado", "Leite & Brazdil 2005 ICML"),
}

# base (X.csv) columns -> (group, family label in Portuguese, reference)
BASE_COLUMN_INFO = {}


def _add_base(cols, group, family_pt, reference):
    for c in cols:
        BASE_COLUMN_INFO[c] = (group, family_pt, reference)


_add_base(
    ["n_instances", "n_classes", "log_instances", "instances_per_class"],
    "simple", "simples", "descritores de tamanho do dataset (hs_meta_features.py)",
)
_add_base(
    ["majority_pct", "minority_pct", "imbalance_ratio", "class_entropy"],
    "balance", "balanceamento", "entropia de Shannon da distribuição de classes (hs_meta_features.py)",
)
_add_base(
    ["vocab_size", "log_vocab_size", "vocab_per_instance", "type_token_ratio", "hapax_ratio",
     "mean_doc_len_words", "std_doc_len_words", "mean_doc_len_chars", "mean_word_len"],
    "lexical", "lexical", "estatísticas lexicais do corpus (hs_meta_features.py)",
)
_add_base(
    ["pct_docs_with_mention", "mean_mentions_per_doc", "pct_docs_with_url", "pct_docs_with_hashtag",
     "mean_hashtags_per_doc", "pct_docs_with_emoji", "uppercase_char_ratio", "exclamation_rate",
     "elongation_rate", "pct_docs_with_profanity", "profanity_token_rate"],
    "social_media", "redes sociais", "descritores de linguagem de redes sociais (hs_meta_features.py)",
)
_add_base(
    ["mean_mutual_info", "max_mutual_info"],
    "info_theory", "teoria da informação", "mutual_info_classif (scikit-learn)",
)
_add_base(
    ["centroid_cosine_gap", "knn1_disagreement", "silhouette_by_label"],
    "separability", "separabilidade", "separabilidade em espaço de embeddings MiniLM (hs_meta_features.py)",
)
_add_base(
    ["is_english", "is_multilingual_row", "is_twitter"],
    "manifest", "metadados", "metadados do manifest do dataset",
)


def _new_group_for_col(col):
    for g in NEW_GROUP_ORDER:
        if col.startswith(g + "_"):
            return g
    raise ValueError(f"unrecognized new feature column {col!r}")


def process_dataset(dataset_id, datasets_dir, embeddings_dir, manifest_row, lexica_dir):
    timings = {}

    df = pd.read_parquet(os.path.join(datasets_dir, f"{dataset_id}.parquet"))
    texts = df["text"].astype(str).tolist()
    labels = df["label"].to_numpy()

    # embedding load time is not counted in any group cost
    minilm_emb = np.load(os.path.join(embeddings_dir, f"{dataset_id}__minilm.npy"))
    hatebert_emb = np.load(os.path.join(embeddings_dir, f"{dataset_id}__hatebert.npy"))
    mbert_emb = np.load(os.path.join(embeddings_dir, f"{dataset_id}__mbert.npy"))

    feats = {}

    t = time.perf_counter()
    feats.update(dmf.compute_tok_features(texts))
    timings["tok"] = time.perf_counter() - t

    t = time.perf_counter()
    lexicon = dmf.get_lexicon_for_language(manifest_row.get("language", ""), lexica_dir)
    feats.update(dmf.compute_hl_features(texts, labels, lexicon))
    timings["hl"] = time.perf_counter() - t

    t = time.perf_counter()
    feats.update(dmf.compute_ch_features(texts, labels))
    timings["ch"] = time.perf_counter() - t

    t = time.perf_counter()
    feats.update(dmf.compute_lm_features(texts, labels, minilm_emb, hatebert_emb, random_state=RANDOM_STATE))
    timings["lm"] = time.perf_counter() - t

    t = time.perf_counter()
    feats.update(dmf.compute_cx_features(minilm_emb, labels, random_state=RANDOM_STATE))
    timings["cx"] = time.perf_counter() - t

    # shared preprocessing for ih_/mt_: label-blind TF-IDF + kNN same-class counts
    t = time.perf_counter()
    tfidf_matrix = dmf.build_hardness_tfidf(texts)
    shared_knn = dmf.compute_shared_knn(minilm_emb, tfidf_matrix, labels, k=5)
    timings["shared_knn"] = time.perf_counter() - t

    t = time.perf_counter()
    feats.update(dmf.compute_ih_features(minilm_emb, hatebert_emb, mbert_emb, tfidf_matrix, labels, shared_knn=shared_knn))
    timings["ih"] = time.perf_counter() - t

    t = time.perf_counter()
    feats.update(dmf.compute_mt_features(labels, shared_knn))
    timings["mt"] = time.perf_counter() - t

    t = time.perf_counter()
    feats.update(dmf.compute_lc_features(texts, labels, minilm_emb, random_state=RANDOM_STATE))
    timings["lc"] = time.perf_counter() - t

    # timing-only recomputation of the base groups (values discarded)
    t = time.perf_counter()
    mf.compute_class_balance_features(labels)
    timings["base_balance"] = time.perf_counter() - t

    t = time.perf_counter()
    mf.compute_text_features(texts)
    timings["base_text"] = time.perf_counter() - t

    t = time.perf_counter()
    mf.compute_mutual_info_features(texts, labels, random_state=RANDOM_STATE)
    timings["base_mutual_info"] = time.perf_counter() - t

    t = time.perf_counter()
    mf.compute_embedding_features(minilm_emb, labels)
    timings["base_embedding"] = time.perf_counter() - t

    t = time.perf_counter()
    mf.compute_manifest_features(manifest_row.get("language", ""), manifest_row.get("platform", ""))
    timings["base_manifest"] = time.perf_counter() - t

    return dataset_id, feats, timings


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets-dir", default=os.path.join(REPO_ROOT, "data_hs", "datasets"))
    ap.add_argument("--manifest", default=os.path.join(REPO_ROOT, "data_hs", "datasets_manifest.csv"))
    ap.add_argument("--embeddings-dir", default=os.path.join(REPO_ROOT, "data_hs", "embeddings"))
    ap.add_argument("--x-csv", default=os.path.join(REPO_ROOT, "data_hs", "X.csv"))
    ap.add_argument("--lexica-dir", default=os.path.join(REPO_ROOT, "data_hs", "lexica"))
    ap.add_argument("--out-dir", default=os.path.join(REPO_ROOT, "data_hs"))
    ap.add_argument("--n-jobs", type=int, default=6)
    ap.add_argument("--max-datasets", type=int, default=None, help="debug: limit number of datasets processed")
    ap.add_argument("--only", default="", help="debug: comma-separated dataset_ids to process")
    args = ap.parse_args()

    def log(msg):
        print(msg, flush=True)

    t_start = time.time()

    X = pd.read_csv(args.x_csv, index_col=0)
    manifest = pd.read_csv(args.manifest, dtype={"dataset_id": str}).set_index("dataset_id", drop=False)

    dataset_ids = [
        d for d in X.index
        if os.path.exists(os.path.join(args.datasets_dir, f"{d}.parquet"))
    ]
    missing_parquet = [d for d in X.index if d not in dataset_ids]
    if missing_parquet:
        log(f"[warn] {len(missing_parquet)} X.csv rows have no parquet file, skipped: {missing_parquet}")

    if args.only:
        wanted = [d.strip() for d in args.only.split(",") if d.strip()]
        unknown = [d for d in wanted if d not in dataset_ids]
        if unknown:
            raise SystemExit(f"--only requested unknown dataset_id(s): {unknown}")
        dataset_ids = [d for d in dataset_ids if d in wanted]
    elif args.max_datasets:
        dataset_ids = dataset_ids[: args.max_datasets]

    log(f"[main] {len(dataset_ids)} datasets to process (n_jobs={args.n_jobs})")

    manifest_rows = {d: manifest.loc[d].to_dict() for d in dataset_ids}

    results = Parallel(n_jobs=args.n_jobs, verbose=10)(
        delayed(process_dataset)(d, args.datasets_dir, args.embeddings_dir, manifest_rows[d], args.lexica_dir)
        for d in dataset_ids
    )

    feats_by_id = {}
    timing_records = []
    for dataset_id, feats, timings in results:
        feats_by_id[dataset_id] = feats
        for group, seconds in timings.items():
            timing_records.append({"dataset_id": dataset_id, "group": group, "seconds": seconds})

    X_new = pd.DataFrame.from_dict(feats_by_id, orient="index")
    X_new.index.name = "dataset_id"
    X_new = X_new.loc[dataset_ids]  # preserve X.csv row order
    X_new = X_new[NEW_COLUMNS]

    # NaN / inf safety net
    X_new = X_new.replace([np.inf, -np.inf], np.nan)
    nan_counts = X_new.isna().sum()
    nan_counts = nan_counts[nan_counts > 0]
    if len(nan_counts):
        log(f"[WARN] NaN/inf found, applying LAST-RESORT median imputation: {dict(nan_counts)}")
        for col, n_missing in nan_counts.items():
            median = X_new[col].median()
            X_new[col] = X_new[col].fillna(median)
            log(f"[WARN]   median-imputed {n_missing} cells in {col!r} (median={median})")

    assert not X_new.isna().any().any(), "NaN remained in X_new after imputation"
    assert np.isfinite(X_new.to_numpy(dtype=float)).all(), "non-finite values remained in X_new"

    dropped_constant = {}
    for col in list(X_new.columns):
        if X_new[col].nunique(dropna=True) <= 1:
            dropped_constant[col] = X_new[col].iloc[0] if len(X_new) else None
            X_new = X_new.drop(columns=[col])
    if dropped_constant:
        log(f"[main] dropped {len(dropped_constant)} constant columns: {list(dropped_constant)}")

    assert list(X_new.index) == dataset_ids, "X_new.index order != processed dataset_ids order"
    if set(dataset_ids) == set(X.index):
        assert list(X_new.index) == list(X.index), "X_new.index != X.csv index (full run)"

    X_extended = pd.concat([X.loc[dataset_ids], X_new], axis=1)

    catalog_rows = []
    for col in X.columns:
        group, family_pt, reference = BASE_COLUMN_INFO[col]
        catalog_rows.append({"feature": col, "origin": "base", "group": group, "family": family_pt, "reference": reference})
    for col in X_new.columns:
        group = _new_group_for_col(col)
        family_pt, reference = NEW_GROUP_INFO[group]
        catalog_rows.append({"feature": col, "origin": "new", "group": group, "family": family_pt, "reference": reference})
    catalog = pd.DataFrame.from_records(catalog_rows)

    os.makedirs(args.out_dir, exist_ok=True)

    def _atomic_write_csv(df_or_records, path, index=True):
        tmp = path + ".tmp"
        if isinstance(df_or_records, pd.DataFrame):
            df_or_records.to_csv(tmp, index=index)
        os.replace(tmp, path)

    extraction_cost = pd.DataFrame.from_records(timing_records)[["dataset_id", "group", "seconds"]]

    _atomic_write_csv(X_new, os.path.join(args.out_dir, "X_new.csv"))
    _atomic_write_csv(X_extended, os.path.join(args.out_dir, "X_extended.csv"))
    _atomic_write_csv(extraction_cost, os.path.join(args.out_dir, "extraction_cost.csv"), index=False)
    _atomic_write_csv(catalog, os.path.join(args.out_dir, "meta_feature_catalog.csv"), index=False)

    total_time = time.time() - t_start

    log("\n===== REPORT =====")
    log(f"X_new shape: {X_new.shape}, X_extended shape: {X_extended.shape}")
    log(f"Dropped constant columns: {list(dropped_constant)}")
    log(f"Total wall-clock: {total_time:.1f}s")

    log("\nPer-group cost (total seconds, mean seconds/dataset):")
    grp = extraction_cost.groupby("group")["seconds"].agg(["sum", "mean", "count"]).sort_values("sum", ascending=False)
    for group, row in grp.iterrows():
        log(f"  {group:16s} total={row['sum']:8.2f}s  mean/dataset={row['mean']:7.3f}s  n={int(row['count'])}")

    log("\nTop-10 slowest (dataset, group):")
    top10 = extraction_cost.sort_values("seconds", ascending=False).head(10)
    for _, row in top10.iterrows():
        log(f"  {row['dataset_id']:40s} {row['group']:14s} {row['seconds']:8.3f}s")

    log("\nColumn value ranges (new features):")
    for col in X_new.columns:
        s = X_new[col]
        log(f"  {col:40s} min={s.min():.4f} median={s.median():.4f} max={s.max():.4f}")
    log("===================")


if __name__ == "__main__":
    main()
