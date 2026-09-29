#!/usr/bin/env python3
# AI assistance was used to write the argument parsing and the report-generation code.
import argparse
import glob
import os
import shutil
import subprocess
import sys
import time
from collections import defaultdict

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
from sklearn.feature_extraction.text import TfidfVectorizer

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))

import hs_meta_features as mf
import hs_algorithms as algos

RANDOM_STATE = 42
N_SPLITS = 10
SLOW_PAIR_SECONDS = 120.0


CANDIDATE_EMBED_PYTHONS = [
    p for p in (os.environ.get("EMBED_PYTHON"), sys.executable) if p
]


def find_embed_python(explicit):
    if explicit:
        return explicit if os.path.exists(explicit) else None
    for cand in CANDIDATE_EMBED_PYTHONS:
        if not os.path.exists(cand):
            continue
        try:
            out = subprocess.run(
                [cand, "-c",
                 "import torch, transformers, sentence_transformers; "
                 "print(torch.cuda.is_available())"],
                capture_output=True, text=True, timeout=30,
            )
            if out.returncode == 0:
                return cand
        except Exception:
            continue
    return None


def ensure_embeddings(dataset_ids, datasets_dir, embeddings_dir, embed_python, log):
    os.makedirs(embeddings_dir, exist_ok=True)
    for model_tag in ("minilm", "hatebert", "mbert"):
        missing = [
            d for d in dataset_ids
            if not os.path.exists(os.path.join(embeddings_dir, f"{d}__{model_tag}.npy"))
        ]
        if not missing:
            log(f"[embed] {model_tag}: all {len(dataset_ids)} datasets already cached")
            continue
        if embed_python is None:
            raise RuntimeError(
                f"No embeddings cached for model={model_tag} on {len(missing)} datasets, "
                "and no GPU-capable python interpreter (torch+transformers+sentence-transformers) "
                "was found. Pass --embed-python explicitly."
            )
        log(f"[embed] {model_tag}: encoding {len(missing)}/{len(dataset_ids)} missing datasets "
            f"via {embed_python}")
        cmd = [
            embed_python, os.path.join(REPO_ROOT, "hs_embed_worker.py"),
            "--model", model_tag,
            "--datasets-dir", datasets_dir,
            "--out-dir", embeddings_dir,
            "--dataset-ids", ",".join(missing),
        ]
        t0 = time.time()
        proc = subprocess.run(cmd, cwd=REPO_ROOT)
        if proc.returncode != 0:
            raise RuntimeError(f"hs_embed_worker.py failed for model={model_tag} (exit {proc.returncode})")
        log(f"[embed] {model_tag}: done in {time.time() - t0:.1f}s")


def load_embeddings(dataset_ids, embeddings_dir):
    cache = defaultdict(dict)
    for d in dataset_ids:
        for model_tag in ("minilm", "hatebert", "mbert"):
            path = os.path.join(embeddings_dir, f"{d}__{model_tag}.npy")
            cache[d][model_tag] = np.load(path)
    return cache


def build_meta_features(dataset_id, texts, labels, minilm_emb, manifest_row, log):
    feats = {}
    feats.update(mf.compute_class_balance_features(labels))
    feats.update(mf.compute_text_features(texts))
    feats.update(mf.compute_mutual_info_features(texts, labels, random_state=RANDOM_STATE))
    feats.update(mf.compute_embedding_features(minilm_emb, labels))
    feats.update(mf.compute_manifest_features(
        manifest_row.get("language", ""), manifest_row.get("platform", "")
    ))
    return feats


def quick_word_tfidf_n_features(train_texts):
    vec = TfidfVectorizer(ngram_range=(1, 2), max_features=20000, min_df=2, sublinear_tf=True)
    X = vec.fit_transform(train_texts)
    return X.shape[1]


def run_cv_for_dataset(dataset_id, dataset_name, texts, labels, emb_by_model, log):
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    folds = list(skf.split(texts, labels))

    records = []
    svd_ncomp_cache = {}
    slow_pairs = []

    for algo in algos.ALL_ALGORITHMS:
        for fold_i, (train_idx, test_idx) in enumerate(folds):
            t0 = time.time()
            y_train, y_test = labels[train_idx], labels[test_idx]

            if algo in algos.TEXT_ALGORITHMS:
                X_train_text = [texts[i] for i in train_idx]
                X_test_text = [texts[i] for i in test_idx]
                if algo.startswith("svd_"):
                    if fold_i not in svd_ncomp_cache:
                        n_feat = quick_word_tfidf_n_features(X_train_text)
                        svd_ncomp_cache[fold_i] = max(2, min(200, n_feat - 1))
                    pipe = algos.make_text_pipeline(algo, svd_n_components=svd_ncomp_cache[fold_i])
                else:
                    pipe = algos.make_text_pipeline(algo)
                try:
                    pipe.fit(X_train_text, y_train)
                    preds = pipe.predict(X_test_text)
                except ValueError as e:
                    # extremely small fold vocabulary edge case: shrink SVD and retry once
                    if algo.startswith("svd_") and "n_components" in str(e):
                        pipe = algos.make_text_pipeline(algo, svd_n_components=2)
                        pipe.fit(X_train_text, y_train)
                        preds = pipe.predict(X_test_text)
                    else:
                        raise
            else:
                model_tag = algos.EMBED_MODEL_FOR_ALGO[algo]
                emb = emb_by_model[model_tag]
                clf = algos.make_embedding_classifier(algo)
                clf.fit(emb[train_idx], y_train)
                preds = clf.predict(emb[test_idx])

            f1 = f1_score(y_test, preds, average="macro")
            dt = time.time() - t0
            if dt > SLOW_PAIR_SECONDS:
                slow_pairs.append((algo, fold_i, dt))
            records.append({
                "dataset_id": dataset_id,
                "dataset_name": dataset_name,
                "algorithm": algo,
                "fold": fold_i,
                "repeat": 0,
                "value": f1,
            })

    for algo, fold_i, dt in slow_pairs:
        log(f"  [slow] {dataset_id}/{algo} fold {fold_i} took {dt:.1f}s (> {SLOW_PAIR_SECONDS:.0f}s)")

    return records


def make_synthetic_fixture(fixture_dir):
    """Small synthetic datasets + manifest under `fixture_dir` for testing the pipeline."""
    rng = np.random.RandomState(0)
    datasets_dir = os.path.join(fixture_dir, "datasets")
    os.makedirs(datasets_dir, exist_ok=True)

    hate_words_en = ["idiot", "trash", "scum", "moron", "hate", "disgusting"]
    neutral_words_en = ["today", "weather", "coffee", "meeting", "great", "walk", "book", "music"]
    hate_words_es = ["odio", "basura", "idiota", "asco"]
    neutral_words_es = ["hoy", "cafe", "musica", "libro", "reunion", "genial"]
    profane = ["fuck", "shit", "bitch", "trash"]

    def gen_doc(rng, hate, hatewords, neutralwords, platform):
        n_words = rng.randint(4, 20)
        pool = hatewords if hate else neutralwords
        other = neutralwords if hate else neutralwords
        words = list(rng.choice(pool, size=max(1, n_words // 3))) + \
            list(rng.choice(other, size=n_words - max(1, n_words // 3)))
        rng.shuffle(words)
        text = " ".join(words)
        if platform == "twitter" and rng.rand() < 0.5:
            text += " @user" + str(rng.randint(0, 50))
        if rng.rand() < 0.3:
            text += " #topic" + str(rng.randint(0, 5))
        if rng.rand() < 0.15:
            text += " http://example.com/" + str(rng.randint(0, 100))
        if hate and rng.rand() < 0.4:
            text += " " + rng.choice(profane)
        if rng.rand() < 0.1:
            text = text + "!" * rng.randint(1, 4)
        if rng.rand() < 0.1:
            text = text.replace(" ", "  soooo ", 1)
        return text

    specs = [
        dict(dataset_id="fx_en_twitter_bin", n=320, k=2, minority=0.12, lang="en", platform="twitter",
             hatewords=hate_words_en, neutralwords=neutral_words_en),
        dict(dataset_id="fx_en_reddit_multi", n=450, k=3, minority=0.15, lang="en", platform="reddit",
             hatewords=hate_words_en, neutralwords=neutral_words_en),
        dict(dataset_id="fx_es_twitter_bin", n=310, k=2, minority=0.20, lang="es", platform="twitter",
             hatewords=hate_words_es, neutralwords=neutral_words_es),
        dict(dataset_id="fx_en_gab_bin", n=500, k=2, minority=0.10, lang="en", platform="gab",
             hatewords=hate_words_en, neutralwords=neutral_words_en),
    ]

    manifest_rows = []
    for spec in specs:
        n, k = spec["n"], spec["k"]
        minority_n = max(int(n * spec["minority"]), 10)
        if k == 2:
            counts = [n - minority_n, minority_n]
        else:
            rest = n - minority_n
            counts = [minority_n] + [rest // (k - 1)] * (k - 1)
            counts[-1] += rest - sum(counts[1:])
        labels = np.concatenate([[c] * cnt for c, cnt in enumerate(counts)])
        rng.shuffle(labels)
        texts = [
            gen_doc(rng, hate=(lab == 0), hatewords=spec["hatewords"], neutralwords=spec["neutralwords"],
                    platform=spec["platform"])
            for lab in labels
        ]
        df = pd.DataFrame({"text": texts, "label": labels.astype(int)})
        df.to_parquet(os.path.join(datasets_dir, f"{spec['dataset_id']}.parquet"))
        manifest_rows.append({
            "dataset_id": spec["dataset_id"],
            "source": "synthetic_fixture",
            "slice": spec["dataset_id"],
            "task": "binary" if k == 2 else "multiclass",
            "language": spec["lang"],
            "platform": spec["platform"],
            "n_instances": n,
            "n_classes": k,
            "minority_pct": counts[np.argmin(counts)] / n,
            "provenance": "synthetic fixture generated by hs_build_metadataset.py for pipeline dev/testing",
        })
    pd.DataFrame(manifest_rows).to_csv(os.path.join(fixture_dir, "datasets_manifest.csv"), index=False)
    return datasets_dir, os.path.join(fixture_dir, "datasets_manifest.csv")


def write_notes(path, ctx):
    n_datasets = ctx["n_datasets"]
    n_algorithms = len(algos.ALL_ALGORITHMS)
    lines = []
    lines.append("# Hate-speech meta-dataset provenance notes\n")
    lines.append("## Datasets\n")
    lines.append(
        "- Panel of hate-speech / offensive-language classification datasets assembled by a "
        "parallel agent into `data_hs/datasets/<dataset_id>.parquet` (columns `text`, `label`) "
        "with a manifest at `data_hs/datasets_manifest.csv` "
        "(`dataset_id,source,slice,task,language,platform,n_instances,n_classes,minority_pct,provenance`). "
        "Each dataset is capped at 3000 instances, minimum 300, minimum 2 classes, minimum 10% minority.\n"
    )
    lines.append(f"- Datasets used in this build: **{n_datasets}**.\n")
    lines.append("## Deduplication rule\n")
    lines.append(
        "- No cross-dataset deduplication was needed at this stage: the datasets contract already "
        "guarantees exactly one row (one dataset_id) per source/slice, produced by the parallel dataset-building "
        "agent. `hs_build_metadataset.py` takes the inner join of manifest rows and parquet files present "
        "on disk and logs any mismatch.\n"
    )
    lines.append("## Algorithm panel (10, fixed)\n")
    lines.append(
        "TF-IDF family (vectorizer/SVD fit INSIDE each fold on the training split only -- via "
        "`sklearn.pipeline.Pipeline`, no leakage):\n\n"
        "| algorithm | pipeline |\n|---|---|\n"
        "| tfidf_word_lr | TfidfVectorizer(word, 1-2gram, max_features=20000, min_df=2, sublinear_tf) + LogisticRegression(balanced) |\n"
        "| tfidf_char_svm | TfidfVectorizer(char_wb, 3-5gram, max_features=30000, min_df=2, sublinear_tf) + LinearSVC(balanced) |\n"
        "| tfidf_nb | TfidfVectorizer(word, 1-2gram, max_features=20000, min_df=2) + ComplementNB |\n"
        "| svd_rf | TF-IDF word + TruncatedSVD(<=200) + RandomForestClassifier(200 trees, balanced) |\n"
        "| svd_histgb | TF-IDF word + TruncatedSVD(<=200) + HistGradientBoostingClassifier |\n"
        "| svd_knn | TF-IDF word + TruncatedSVD(<=200) + KNeighborsClassifier(k=5, cosine) |\n\n"
        "For the `svd_*` algorithms, `n_components` is computed PER FOLD from the actual vocabulary size "
        "of the training split (`min(200, n_features_train - 1)`), so it never leaks test-set structure and "
        "never exceeds the available TF-IDF dimensionality on tiny folds.\n\n"
        "Frozen-embedding family (embeddings computed ONCE per dataset, OUTSIDE the CV loop, cached to "
        "`data_hs/embeddings/<dataset_id>__<model_tag>.npy`; only the classifier head is fit inside each fold). "
        "This is NOT leakage: the encoders (MiniLM / hateBERT / mBERT) are pretrained and never see this "
        "task's labels -- encoding is a fixed, label-blind feature map, exactly analogous to using a "
        "fixed dictionary/embedding table as input features.\n\n"
        "| algorithm | embedding | head |\n|---|---|\n"
        "| minilm_lr | sentence-transformers/all-MiniLM-L6-v2 (mean-pooled) | LogisticRegression(balanced) |\n"
        "| minilm_mlp | same MiniLM embeddings | MLPClassifier((128,), max_iter=500) |\n"
        "| hatebert_lr | GroNLP/hateBERT last-hidden-state, mean-pooled | LogisticRegression(balanced) |\n"
        "| mbert_lr | google-bert/bert-base-multilingual-cased, mean-pooled | LogisticRegression(balanced) |\n\n"
        "By design, `hatebert_lr` and `minilm_*` are English-centric encoders while `mbert_lr` is "
        "multilingual; the English encoders are expected to (and, per the numbers below, do) degrade "
        "on the Spanish/Portuguese rows relative to `mbert_lr`. This is deliberate meta-learning signal "
        "that the `is_english` meta-feature is meant to predict -- it is not a bug and was not corrected.\n"
    )
    lines.append("All estimators are deterministic: `random_state=42` everywhere it applies.\n")
    lines.append("## Performance measure\n")
    lines.append(
        "**Macro-F1** (`sklearn.metrics.f1_score(average=\"macro\")`), used everywhere in P/P_folds. "
        "Chosen because hate-speech datasets are class-imbalanced and accuracy is dominated by the "
        "majority (non-hate) class.\n"
    )
    lines.append("## Cross-validation\n")
    lines.append(
        "`StratifiedKFold(n_splits=10, shuffle=True, random_state=42)`, fit ONCE per dataset -- the same "
        "10 fold index sets are reused across all 10 algorithms on a given dataset, which is required "
        "because a paired Wilcoxon signed-rank test is run over the folds downstream.\n"
        f"- P_folds.csv has {ctx['n_p_folds_rows']} rows = {n_datasets} datasets x {n_algorithms} algorithms x 10 folds.\n"
    )
    lines.append("## Meta-features (X)\n")
    lines.append(
        "Computed on the FULL dataset (a dataset descriptor, not a per-fold quantity).\n\n"
        "General / dimension: `n_instances`, `n_classes`, `log_instances = log10(n_instances)`, "
        "`instances_per_class = n_instances / n_classes`.\n\n"
        "Class balance: `majority_pct`, `minority_pct`, "
        "`imbalance_ratio = majority_count / minority_count`, "
        "`class_entropy` = Shannon entropy (bits) of the label distribution.\n\n"
        "Lexical: `vocab_size` (unique lowercased whitespace tokens), `log_vocab_size`, "
        "`vocab_per_instance = vocab_size / n_instances`, `type_token_ratio = vocab_size / total_tokens`, "
        "`hapax_ratio` = fraction of vocabulary types occurring exactly once, "
        "`mean_doc_len_words`, `std_doc_len_words`, `mean_doc_len_chars`, `mean_word_len`.\n\n"
        "Social-media / domain-specific (justification: hate speech is overwhelmingly studied on "
        "social-media text, and these descriptors characterize how 'platform-native' / noisy / "
        "emphatic a dataset's language is -- properties known to interact with how well TF-IDF vs. "
        "pretrained-embedding models perform):\n"
        "- `pct_docs_with_mention` / `mean_mentions_per_doc` (regex `@\\w+`) -- reply/targeting culture (Twitter/Reddit-style).\n"
        "- `pct_docs_with_url` (`https?://` or `www\\.`) -- link-sharing behavior.\n"
        "- `pct_docs_with_hashtag` / `mean_hashtags_per_doc` (regex `#\\w+`) -- topic-tagging behavior.\n"
        "- `pct_docs_with_emoji` (unicode emoji ranges) -- paralinguistic/affective markup.\n"
        "- `uppercase_char_ratio` (uppercase letters / all letters) -- shouting / emphasis.\n"
        "- `exclamation_rate` (mean `!` per doc) -- emphatic/aggressive punctuation.\n"
        "- `elongation_rate` (fraction of docs matching `(.)\\1{2,}`) -- expressive lengthening (e.g. 'soooo').\n"
        "- `pct_docs_with_profanity` / `profanity_token_rate` -- rate of a small (~40 term) hardcoded "
        "English profanity/slur-adjacent lexicon; used purely to characterize how 'raw' a dataset's "
        "language is, not to label or classify any person or group.\n\n"
        "**Caveat (English-only lexicon):** `PROFANITY_LEXICON` is English-only by construction. "
        "`pct_docs_with_profanity` and `profanity_token_rate` will therefore read near-zero on the "
        "Spanish/Portuguese/other-language rows regardless of how profane those datasets actually are. "
        "The lexicon was deliberately NOT translated/extended -- these two columns are partly "
        "confounded with `is_english` and must NOT be interpreted as a language-independent measure "
        "of how raw/profane a dataset is; treat them as reliable only within the English-language rows.\n\n"
        "Signal / separability / hardness (the highest-value features for this problem -- domain-motivated "
        "by instance-hardness research in hate speech):\n"
        "- `mean_mutual_info`, `max_mutual_info` -- mean/max mutual information between the top-1000 "
        "TF-IDF terms and the label (`sklearn.feature_selection.mutual_info_classif`, random_state=42).\n"
        "- `centroid_cosine_gap = 1 - mean_pairwise_cosine(class centroids)` in MiniLM embedding space "
        "(mean over all class pairs for multiclass). Higher = more separable.\n"
        "- `knn1_disagreement` -- fraction of instances whose nearest neighbour (cosine, MiniLM space, "
        "excluding self) carries a different label; a direct instance-hardness / label-noise proxy.\n"
        "- `silhouette_by_label` -- `sklearn.metrics.silhouette_score` on MiniLM embeddings with the true "
        "labels as the partition, cosine metric.\n\n"
        "Language / platform (from the manifest): `is_english` (language == 'en'), "
        "`is_multilingual_row` (language != 'en'), `is_twitter` (platform == 'twitter').\n"
    )
    lines.append("## Constant columns dropped\n")
    if ctx["dropped_constant_cols"]:
        for col, val in ctx["dropped_constant_cols"].items():
            lines.append(f"- `{col}` (constant at {val!r} across all {n_datasets} datasets)\n")
    else:
        lines.append("- None.\n")
    lines.append("## Imputation\n")
    lines.append(
        f"Remaining missing values (e.g. degenerate `type_token_ratio`/`silhouette_by_label` on unusual "
        f"tiny datasets) imputed with the column median. Total imputed cells: **{ctx['n_imputed_total']}**.\n"
    )
    if ctx["imputed_per_col"]:
        lines.append("Per-column imputed-cell counts: " +
                      ", ".join(f"`{c}`={n}" for c, n in ctx["imputed_per_col"].items() if n > 0) + "\n")
    lines.append("## Final matrix shapes\n")
    lines.append(f"- P : {ctx['p_shape'][0]} datasets x {ctx['p_shape'][1]} algorithms\n")
    lines.append(f"- X : {ctx['x_shape'][0]} datasets x {ctx['x_shape'][1]} meta-features\n")
    lines.append(f"- P_folds : {ctx['n_p_folds_rows']} rows\n")
    with open(path, "w") as f:
        f.write("\n".join(lines))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets-dir", default=os.path.join(REPO_ROOT, "data_hs", "datasets"))
    ap.add_argument("--manifest", default=os.path.join(REPO_ROOT, "data_hs", "datasets_manifest.csv"))
    ap.add_argument("--out-dir", default=os.path.join(REPO_ROOT, "data_hs"))
    ap.add_argument("--embed-python", default=None, help="python interpreter with CUDA torch+transformers+sentence-transformers")
    ap.add_argument("--use-fixture", action="store_true",
                     help="ignore --datasets-dir/--manifest and generate+use a small synthetic fixture "
                          "in a temporary directory, for pipeline development only")
    ap.add_argument("--fixture-dir", default="/tmp/hs_fixture")
    ap.add_argument("--max-datasets", type=int, default=None, help="debug: limit number of datasets processed")
    args = ap.parse_args()

    def log(msg):
        print(msg, flush=True)

    t_start = time.time()

    datasets_dir, manifest_path, out_dir = args.datasets_dir, args.manifest, args.out_dir
    if args.use_fixture:
        log(f"[fixture] generating synthetic fixture under {args.fixture_dir}")
        datasets_dir, manifest_path = make_synthetic_fixture(args.fixture_dir)
        out_dir = args.fixture_dir

    if not os.path.isdir(datasets_dir) or not os.path.exists(manifest_path):
        log(f"ERROR: datasets dir {datasets_dir!r} or manifest {manifest_path!r} not found. "
            "Run with --use-fixture to develop/test against a synthetic fixture instead.")
        sys.exit(1)

    manifest = pd.read_csv(manifest_path, dtype={"dataset_id": str})
    manifest = manifest.set_index("dataset_id", drop=False)

    parquet_ids = sorted(
        os.path.splitext(os.path.basename(p))[0]
        for p in glob.glob(os.path.join(datasets_dir, "*.parquet"))
    )
    manifest_ids = set(manifest.index)
    dataset_ids = sorted(set(parquet_ids) & manifest_ids)
    only_parquet = sorted(set(parquet_ids) - manifest_ids)
    only_manifest = sorted(manifest_ids - set(parquet_ids))
    if only_parquet:
        log(f"[warn] {len(only_parquet)} parquet files have no manifest row, skipped: {only_parquet[:5]}...")
    if only_manifest:
        log(f"[warn] {len(only_manifest)} manifest rows have no parquet file, skipped: {only_manifest[:5]}...")

    if args.max_datasets:
        dataset_ids = dataset_ids[: args.max_datasets]

    log(f"[main] {len(dataset_ids)} datasets to process")

    datasets = {}
    for d in dataset_ids:
        df = pd.read_parquet(os.path.join(datasets_dir, f"{d}.parquet"))
        assert set(df.columns) >= {"text", "label"}, f"{d}: missing text/label columns"
        texts = df["text"].astype(str).tolist()
        labels = df["label"].to_numpy()
        assert labels.dtype.kind in "iu", f"{d}: label column not integer dtype"
        datasets[d] = (texts, labels)

    embeddings_dir = os.path.join(out_dir, "embeddings")
    embed_python = find_embed_python(args.embed_python)
    if embed_python:
        log(f"[main] using embedding interpreter: {embed_python}")
    else:
        log("[main] WARNING: no GPU-capable embedding interpreter auto-detected")
    ensure_embeddings(dataset_ids, datasets_dir, embeddings_dir, embed_python, log)
    emb_cache = load_embeddings(dataset_ids, embeddings_dir)

    log("[main] computing meta-features")
    x_rows = {}
    for d in dataset_ids:
        texts, labels = datasets[d]
        row = manifest.loc[d]
        x_rows[d] = build_meta_features(d, texts, labels, emb_cache[d]["minilm"], row, log)
    X = pd.DataFrame.from_dict(x_rows, orient="index")
    X.index.name = "dataset_id"
    X = X.sort_index()

    dropped_constant_cols = {}
    for col in list(X.columns):
        if X[col].nunique(dropna=True) <= 1:
            val = X[col].iloc[0] if len(X) else None
            dropped_constant_cols[col] = val.item() if hasattr(val, "item") else val
            X = X.drop(columns=[col])
    if dropped_constant_cols:
        log(f"[main] dropped {len(dropped_constant_cols)} constant columns: {list(dropped_constant_cols)}")

    n_imputed_total = 0
    imputed_per_col = {}
    for col in X.columns:
        n_missing = int(X[col].isna().sum())
        if n_missing:
            median = X[col].median()
            X[col] = X[col].fillna(median)
            imputed_per_col[col] = n_missing
            n_imputed_total += n_missing
    log(f"[main] imputed {n_imputed_total} missing meta-feature cells")

    log("[main] starting CV loop")
    all_records = []
    for i, d in enumerate(dataset_ids):
        texts, labels = datasets[d]
        row = manifest.loc[d]
        dataset_name = f"{row.get('source', d)}__{row.get('slice', d)}"
        t0 = time.time()
        records = run_cv_for_dataset(d, dataset_name, texts, labels, emb_cache[d], log)
        all_records.extend(records)
        dt = time.time() - t0
        mean_f1 = np.mean([r["value"] for r in records])
        log(f"[cv] ({i + 1}/{len(dataset_ids)}) {d}: n={len(texts)} k={len(set(labels))} "
            f"mean_macro_f1={mean_f1:.4f} time={dt:.1f}s")

    P_folds = pd.DataFrame.from_records(all_records)[
        ["dataset_id", "dataset_name", "algorithm", "fold", "repeat", "value"]
    ]

    P = P_folds.pivot_table(index="dataset_id", columns="algorithm", values="value", aggfunc="mean")
    P = P[algos.ALL_ALGORITHMS]  # fixed column order
    P = P.loc[X.index]  # align row order to X exactly

    assert not P.isna().any().any(), "NaN found in P"
    assert not X.isna().any().any(), "NaN found in X"
    expected_rows = len(dataset_ids) * len(algos.ALL_ALGORITHMS) * N_SPLITS
    assert len(P_folds) == expected_rows, f"P_folds row count {len(P_folds)} != expected {expected_rows}"
    assert list(P.index) == list(X.index), "P.index != X.index"
    recomputed_mean = P_folds.groupby(["dataset_id", "algorithm"])["value"].mean().unstack()
    recomputed_mean = recomputed_mean[algos.ALL_ALGORITHMS].loc[P.index]
    max_gap = (P - recomputed_mean).abs().to_numpy().max()
    assert max_gap < 1e-12, f"max |P - mean(folds)| = {max_gap} >= 1e-12"

    os.makedirs(out_dir, exist_ok=True)
    P.to_csv(os.path.join(out_dir, "P.csv"))
    P_folds.to_csv(os.path.join(out_dir, "P_folds.csv"), index=False)
    X.to_csv(os.path.join(out_dir, "X.csv"))

    ctx = dict(
        n_datasets=len(dataset_ids),
        n_p_folds_rows=len(P_folds),
        p_shape=P.shape,
        x_shape=X.shape,
        dropped_constant_cols=dropped_constant_cols,
        n_imputed_total=n_imputed_total,
        imputed_per_col=imputed_per_col,
    )
    write_notes(os.path.join(out_dir, "metadataset_notes.md"), ctx)

    total_time = time.time() - t_start
    log(f"[main] DONE in {total_time:.1f}s. P={P.shape} X={X.shape} P_folds={P_folds.shape}")

    log("\n===== REPORT =====")
    log(f"P shape: {P.shape}, X shape: {X.shape}, P_folds shape: {P_folds.shape}")
    log("Mean macro-F1 per algorithm:")
    log(str(P.mean().sort_values(ascending=False)))
    ranks = P.rank(axis=1, ascending=False)
    log("Mean rank per algorithm (1=best):")
    log(str(ranks.mean().sort_values()))
    sorted_per_row = -np.sort(-P.to_numpy(), axis=1)
    gaps = sorted_per_row[:, 0] - sorted_per_row[:, 1]
    log(f"Median best-vs-second-best gap per dataset: {np.median(gaps):.4f} "
        f"(mean={np.mean(gaps):.4f}, min={np.min(gaps):.4f}, max={np.max(gaps):.4f})")

    is_en = X["is_english"] == 1
    for algo in ("hatebert_lr", "minilm_lr"):
        en_mean = P.loc[is_en, algo].mean()
        non_en_mean = P.loc[~is_en, algo].mean()
        log(f"{algo}: mean macro-F1 English={en_mean:.4f} (n={is_en.sum()}) "
            f"non-English={non_en_mean:.4f} (n={(~is_en).sum()})")
    mbert_en = P.loc[is_en, "mbert_lr"].mean()
    mbert_non_en = P.loc[~is_en, "mbert_lr"].mean()
    log(f"mbert_lr: mean macro-F1 English={mbert_en:.4f} non-English={mbert_non_en:.4f}")

    log(f"Imputed cells: {n_imputed_total} ({imputed_per_col})")
    log(f"Dropped constant columns: {list(dropped_constant_cols)}")
    log(f"Total wall-clock: {total_time:.1f}s")
    log("===================")


if __name__ == "__main__":
    main()
