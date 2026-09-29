import os
import re
import sys
import time
import traceback
import warnings
from collections import Counter

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")


SEED = 42
CAP = 3000
MIN_TOTAL = 300
MIN_FRAC = 0.10
MIN_CLASS_N = 30

ROOT = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(ROOT, "data_hs")
DATASETS_DIR = os.path.join(OUT_DIR, "datasets")
MANIFEST_PATH = os.path.join(OUT_DIR, "datasets_manifest.csv")
NOTES_PATH = os.path.join(OUT_DIR, "datasets_notes.md")


TCC = os.environ.get("TCC_ROOT", os.path.join(ROOT, "fontes_locais"))

np.random.seed(SEED)

MANIFEST_ROWS = []
NOTES = []
FAIL_NOTES = [] 


def log(msg):
    print(msg)
    NOTES.append(msg)


def log_fail(msg):
    log(msg)
    FAIL_NOTES.append(msg)


def clean_text_label(df, text_col, label_col):
    """Drop null/empty text, coerce to str, strip, dedup exact match on text."""
    df = df[[text_col, label_col]].copy()
    df.columns = ["text", "label_raw"]
    df["text"] = df["text"].astype(str).str.strip()
    df = df[df["text"].str.len() > 0]
    df = df[df["text"].str.lower() != "nan"]
    df = df.dropna(subset=["label_raw"])
    df = df.drop_duplicates(subset=["text"], keep="first")
    return df.reset_index(drop=True)


def rebalance_and_cap(df, tag):
    """Apply the hard rules; returns the final df or None if dropped (see log_fail)."""
    if df is None or len(df) == 0:
        log_fail(f"{tag}: DROPPED - empty after cleaning")
        return None

    counts = df["label_raw"].value_counts()
    keep = counts[counts >= MIN_CLASS_N].index
    if len(keep) < len(counts):
        dropped = sorted(set(counts.index) - set(keep))
        log(f"{tag}: dropping classes with < {MIN_CLASS_N} instances: {dropped}")
    df = df[df["label_raw"].isin(keep)].copy()
    counts = df["label_raw"].value_counts()

    if len(counts) < 2:
        log_fail(f"{tag}: DROPPED - fewer than 2 classes with >= {MIN_CLASS_N} "
                  f"instances after cleaning (counts={counts.to_dict()})")
        return None

    min_class = counts.idxmin()
    min_count = int(counts.min())
    total = int(counts.sum())

    if min_count / total < MIN_FRAC:
        # undersample the other classes so the minority reaches MIN_FRAC
        other_counts = counts.drop(min_class)
        other_total = int(other_counts.sum())
        allowed_other_total = int(min_count * (1 - MIN_FRAC) / MIN_FRAC)
        allowed_other_total = min(allowed_other_total, other_total)
        scale = allowed_other_total / other_total if other_total > 0 else 0.0

        parts = [df[df["label_raw"] == min_class]]
        for cls, cnt in other_counts.items():
            n_target = int(round(cnt * scale))
            n_target = max(0, min(n_target, cnt))
            sub = df[df["label_raw"] == cls]
            if n_target < cnt:
                sub = sub.sample(n=n_target, random_state=SEED)
            if len(sub) > 0:
                parts.append(sub)
        df = pd.concat(parts, ignore_index=True)
        log(f"{tag}: rebalanced (undersampled majority classes) so minority "
            f"class '{min_class}' reaches >= {MIN_FRAC:.0%}")

    # re-check class-size floor: rebalancing could have shrunk a non-anchor class below MIN_CLASS_N
    counts = df["label_raw"].value_counts()
    keep = counts[counts >= MIN_CLASS_N].index
    if len(keep) < len(counts):
        df = df[df["label_raw"].isin(keep)].copy()
        counts = df["label_raw"].value_counts()
    if len(counts) < 2:
        log_fail(f"{tag}: DROPPED - fewer than 2 classes survive rebalancing")
        return None

    if len(df) < MIN_TOTAL:
        log_fail(f"{tag}: DROPPED - only {len(df)} instances after balancing "
                  f"(< {MIN_TOTAL} minimum)")
        return None

    # cap at CAP via stratified split (preserves the now-fixed class proportions)
    if len(df) > CAP:
        y = df["label_raw"].values
        try:
            df, _ = train_test_split(
                df, train_size=CAP, stratify=y, random_state=SEED
            )
        except ValueError as e:
            log(f"{tag}: stratified cap failed ({e}); falling back to "
                f"per-class proportional sampling")
            frac_keep = CAP / len(df)
            parts = []
            for cls, g in df.groupby("label_raw"):
                n = max(1, int(round(len(g) * frac_keep)))
                n = min(n, len(g))
                parts.append(g.sample(n=n, random_state=SEED))
            df = pd.concat(parts, ignore_index=True)
        df = df.reset_index(drop=True)

    # Final validation against every hard rule.
    counts = df["label_raw"].value_counts()
    if len(df) < MIN_TOTAL:
        log_fail(f"{tag}: DROPPED post-cap - {len(df)} < {MIN_TOTAL}")
        return None
    if len(counts) < 2:
        log_fail(f"{tag}: DROPPED post-cap - fewer than 2 classes")
        return None
    if counts.min() < MIN_CLASS_N:
        log_fail(f"{tag}: DROPPED post-cap - smallest class {counts.min()} "
                  f"< {MIN_CLASS_N}")
        return None
    if counts.min() / len(df) < MIN_FRAC - 1e-9:
        log_fail(f"{tag}: DROPPED post-cap - minority frac "
                  f"{counts.min() / len(df):.3f} < {MIN_FRAC}")
        return None

    return df.reset_index(drop=True)


def finalize(raw_df, text_col, label_col, *,
             dataset_id, source, slice_desc, task,
             language, platform, provenance):
    """Clean, rebalance/cap, save parquet and append a manifest row. Returns True if written."""
    tag = dataset_id
    try:
        if raw_df is None:
            log_fail(f"{tag}: DROPPED - loader returned no data")
            return False
        df = clean_text_label(raw_df, text_col, label_col)
        df = rebalance_and_cap(df, tag)
        if df is None:
            return False

        classes_sorted = sorted(df["label_raw"].unique(), key=lambda x: str(x))
        mapping = {c: i for i, c in enumerate(classes_sorted)}
        out = pd.DataFrame({
            "text": df["text"].values,
            "label": df["label_raw"].map(mapping).astype(int).values,
        })

        out_path = os.path.join(DATASETS_DIR, f"{dataset_id}.parquet")
        out.to_parquet(out_path, index=False)

        n = len(out)
        n_classes = out["label"].nunique()
        minority_pct = out["label"].value_counts(normalize=True).min()

        MANIFEST_ROWS.append({
            "dataset_id": dataset_id,
            "source": source,
            "slice": slice_desc,
            "task": task,
            "language": language,
            "platform": platform,
            "n_instances": n,
            "n_classes": n_classes,
            "minority_pct": round(float(minority_pct), 4),
            "provenance": provenance,
        })
        log(f"{tag}: OK n={n} n_classes={n_classes} "
            f"minority_pct={minority_pct:.3f} classes={classes_sorted}")
        return True
    except Exception as e:
        log_fail(f"{tag}: FAILED with exception - {type(e).__name__}: {e}")
        log(traceback.format_exc())
        return False


def build_ihc():
    base = os.path.join(TCC, "implicit-hate-corpus")
    stg1_path = os.path.join(base, "implicit_hate_v1_stg1_posts.tsv")
    stg2_path = os.path.join(base, "implicit_hate_v1_stg2_posts.tsv")
    stg3_path = os.path.join(base, "implicit_hate_v1_stg3_posts.tsv")

    if not (os.path.exists(stg1_path) and os.path.exists(stg2_path)
            and os.path.exists(stg3_path)):
        log_fail("ihc_*: DROPPED - IHC tsv files not found at expected path")
        return

    stg1 = pd.read_csv(stg1_path, sep="\t")
    stg2 = pd.read_csv(stg2_path, sep="\t")
    stg3 = pd.read_csv(stg3_path, sep="\t")
    prov1 = f"local file {stg1_path} (cols post,class; 21480 rows)"
    prov2 = f"local file {stg2_path} (cols post,implicit_class; 6346 rows)"
    prov3 = f"local file {stg3_path} (cols post,target,implied_statement; 6359 rows)"

    # 1. 3-way (not_hate / implicit_hate / explicit_hate)
    finalize(stg1, "post", "class", dataset_id="ihc_stg1_3way", source="ihc",
              slice_desc="all", task="implicit_explicit_not_3way", language="en",
              platform="twitter",
              provenance=f"{prov1}; explicit_hate is a small minority class, "
                         f"rebalanced by undersampling not_hate/implicit_hate "
                         f"to reach the 10% minority floor.")

    # 2. binary hate_vs_not (implicit+explicit vs not_hate)
    s = stg1.copy()
    s["bin"] = np.where(s["class"] == "not_hate", "not_hate", "hate")
    finalize(s, "post", "bin", dataset_id="ihc_stg1_binary", source="ihc",
              slice_desc="all", task="hate_vs_not", language="en", platform="twitter",
              provenance=f"{prov1}; class collapsed to hate (implicit+explicit) vs not_hate.")

    # 3. implicit_vs_not (drop explicit rows)
    s = stg1[stg1["class"] != "explicit_hate"]
    finalize(s, "post", "class", dataset_id="ihc_stg1_implicit_vs_not", source="ihc",
              slice_desc="explicit_hate rows dropped", task="implicit_vs_not",
              language="en", platform="twitter",
              provenance=f"{prov1}; explicit_hate rows dropped, implicit_hate vs not_hate only.")

    # 4. explicit_vs_not (drop implicit rows)
    s = stg1[stg1["class"] != "implicit_hate"]
    finalize(s, "post", "class", dataset_id="ihc_stg1_explicit_vs_not", source="ihc",
              slice_desc="implicit_hate rows dropped", task="explicit_vs_not",
              language="en", platform="twitter",
              provenance=f"{prov1}; implicit_hate rows dropped, explicit_hate vs not_hate; "
                         f"explicit_hate is a small minority, rebalanced by undersampling not_hate.")

    # 5. stg2 6-way implicit category (drop 'other')
    s2 = stg2[stg2["implicit_class"] != "other"]
    finalize(s2, "post", "implicit_class", dataset_id="ihc_stg2_6way", source="ihc",
              slice_desc="'other' category dropped", task="implicit_class_6way",
              language="en", platform="twitter",
              provenance=f"{prov2}; the 'other' category (80 rows) dropped, "
                         f"6 remaining implicit categories all naturally >= 10%.")

    # 6-11. one-vs-rest per implicit category
    categories = ["white_grievance", "incitement", "inferiority", "irony",
                  "stereotypical", "threatening"]
    for cat in categories:
        s = s2.copy()
        s["ovr"] = np.where(s["implicit_class"] == cat, cat, "rest")
        finalize(s, "post", "ovr", dataset_id=f"ihc_stg2_{cat}_vs_rest", source="ihc",
                  slice_desc=f"implicit_class={cat} vs rest", task=f"{cat}_vs_rest",
                  language="en", platform="twitter",
                  provenance=f"{prov2}; one-vs-rest over the 6-way implicit category ({cat} vs "
                             f"the other 5 categories combined, 'other' excluded).")

    # 12-17. target-group slices from stg3, joined against stg1 not_hate as negatives
    target_lower = stg3["target"].astype(str).str.lower()
    not_hate_texts = stg1.loc[stg1["class"] == "not_hate", "post"].dropna().astype(str)
    not_hate_texts = not_hate_texts.drop_duplicates()
    groups = {
        "black": r"\bblack",
        "jewish": r"\bjew",
        "muslim": r"\bmuslim|islamic",
        "immigrant": r"immigra|illegal",
        "women": r"\bwomen\b|\bwoman\b|\bfemale",
        "lgbt": r"\bgay|lesbian|lgbt|homosexual|transgender|\btrans\b",
    }
    for grp, pattern in groups.items():
        mask = target_lower.str.contains(pattern, regex=True, na=False)
        pos_texts = stg3.loc[mask, "post"].dropna().astype(str).drop_duplicates()
        n_pos = len(pos_texts)
        if n_pos < MIN_CLASS_N:
            log_fail(f"ihc_target_{grp}_vs_not: DROPPED - only {n_pos} positive "
                      f"instances (< {MIN_CLASS_N})")
            continue
        n_neg = min(len(not_hate_texts), n_pos * 4)
        neg_texts = not_hate_texts.sample(n=n_neg, random_state=SEED)
        df = pd.concat([
            pd.DataFrame({"text": pos_texts, "label": f"target_{grp}"}),
            pd.DataFrame({"text": neg_texts.values, "label": "not_hate"}),
        ], ignore_index=True)
        finalize(df, "text", "label", dataset_id=f"ihc_target_{grp}_vs_not", source="ihc",
                  slice_desc=f"target~={grp} (stg3, regex-normalized) vs stg1 not_hate",
                  task=f"hate_target_{grp}_vs_not", language="en", platform="twitter",
                  provenance=f"{prov3} joined with {prov1} not_hate as negatives; "
                             f"target free text normalized to '{grp}' via regex '{pattern}' "
                             f"over the 'target' column ({n_pos} matched posts).")


def build_hateval():
    """HatEval (SemEval-2019 Task 5) from the HF mirror `valeriobasile/HatEval` (en+es, HS/TR/AG)."""
    import glob
    hits = glob.glob(os.path.expanduser(
        "~/.cache/huggingface/hub/datasets--valeriobasile--HatEval/snapshots/*/data"))
    df = None
    if hits:
        base = hits[0]
        try:
            dfs = [pd.read_parquet(os.path.join(base, f"{split}-00000-of-00001.parquet"))
                   for split in ("train", "dev", "test")]
            df = pd.concat(dfs, ignore_index=True)
        except Exception as e:
            log(f"hateval: local HF cache read failed ({e}), trying datasets.load_dataset")
    if df is None:
        try:
            from datasets import load_dataset
            ds = load_dataset("valeriobasile/HatEval")
            df = pd.concat([ds[s].to_pandas() for s in ds.keys()], ignore_index=True)
        except Exception as e:
            log_fail(f"hateval_*: DROPPED - could not load valeriobasile/HatEval "
                      f"locally or via network: {e}")
            return

    prov = ("HF dataset valeriobasile/HatEval (official SemEval-2019 Task 5 "
            "train+dev+test, found pre-cached at "
            "~/.cache/huggingface/hub/datasets--valeriobasile--HatEval), "
            "split by `language` column.")
    for lang in ["en", "es"]:
        sub = df[df["language"] == lang]
        for label_col, task_name in [("HS", "hate_vs_not"), ("TR", "target_range_ind_vs_gen"),
                                       ("AG", "aggressiveness_binary")]:
            finalize(sub, "text", label_col, dataset_id=f"hateval_{lang}_{label_col}",
                      source="hateval", slice_desc=f"lang={lang}", task=task_name,
                      language=lang, platform="twitter",
                      provenance=f"{prov} lang={lang}, label={label_col}.")


def build_sbic_local():
    base = os.path.join(TCC, "sbic")
    paths = {
        "train": os.path.join(base, "sbic_terms_clean_train_offall.csv"),
        "dev": os.path.join(base, "sbic_terms_clean_dev_offall.csv"),
        "test": os.path.join(base, "sbic_terms_clean_test_offall.csv"),
    }
    if not all(os.path.exists(p) for p in paths.values()):
        log_fail("sbic_offensive_binary: DROPPED - local sbic_terms_clean files not found")
        return
    dfs = [pd.read_csv(p) for p in paths.values()]
    df = pd.concat(dfs, ignore_index=True)
    finalize(df, "text", "label", dataset_id="sbic_offensive_binary", source="sbic",
              slice_desc="all (train+dev+test concatenated)", task="offensive_binary",
              language="en", platform="mixed",
              provenance=f"local files {list(paths.values())} (cols id,text,label,...; "
                         f"label=offensiveYN binarized); platform is 'mixed' because the "
                         f"original SBIC reddit/twitter/gab/stormfront source column was "
                         f"not retained in this term-annotated cut.")


def build_hatewic():
    path = os.path.join(TCC, "hatewic", "HateWiC_IndividualAnnos_with_def.csv")
    if not os.path.exists(path):
        log_fail("hatewic_all: DROPPED - file not found")
        return
    # file is ';'-delimited
    df = pd.read_csv(path, sep=";")
    df = df.dropna(subset=["binary_label"])
    g = df.groupby("example_id").agg(text=("example", "first"),
                                      lab_mean=("binary_label", "mean"))
    g = g[g["lab_mean"] != 0.5]  # drop exact ties (no majority)
    g["label"] = np.where(g["lab_mean"] > 0.5, "hateful", "not_hateful")
    finalize(g.reset_index(), "text", "label", dataset_id="hatewic_all", source="hatewic",
              slice_desc="all, majority-voted per example", task="word_in_context_hateful_vs_not",
              language="en", platform="unknown",
              provenance=f"local file {path} (';'-delimited; individual annotator rows "
                         f"aggregated to example_id level by majority vote of binary_label; "
                         f"exact 0.5 ties dropped). Word-in-context hate annotation, not "
                         f"social-media text.")


def _hf_load(repo, config=None, **kw):
    from datasets import load_dataset
    if config:
        return load_dataset(repo, config, **kw)
    return load_dataset(repo, **kw)


def build_sbic_hf():
    """SBIC fields (whoTarget, intentYN, sexYN, ...) from allenai/social_bias_frames."""
    tag_group = "sbic_hf_*"
    try:
        ds = _hf_load("allenai/social_bias_frames", revision="refs/convert/parquet")
        df = pd.concat([ds[s].to_pandas() for s in ds.keys()], ignore_index=True)
    except Exception as e:
        log_fail(f"{tag_group}: DROPPED - allenai/social_bias_frames failed to load: "
                  f"{type(e).__name__}: {e}")
        return

    prov_base = ("HF dataset allenai/social_bias_frames (loaded via "
                 "revision=refs/convert/parquet, since the dataset-script form is "
                 "no longer supported by the `datasets` library), train+validation+test "
                 "concatenated.")

    def clean01(s):
        s = pd.to_numeric(s, errors="coerce")
        return s

    df["offensiveYN_n"] = clean01(df["offensiveYN"])
    df["sexYN_n"] = clean01(df["sexYN"])
    df["whoTarget_n"] = clean01(df["whoTarget"])

    platform_map = {
        "twitter": lambda ds_: ds_.isin(["t/founta", "t/davidson", "t/waseem"]),
        "reddit": lambda ds_: ds_.str.startswith("r/") | (ds_ == "redditMicroagressions"),
        "stormfront": lambda ds_: ds_ == "Stormfront",
        "gab": lambda ds_: ds_ == "Gab",
    }
    for plat, cond in platform_map.items():
        mask = cond(df["dataSource"].astype(str)) & df["offensiveYN_n"].isin([0.0, 1.0])
        sub = df.loc[mask, ["post", "offensiveYN_n"]]
        finalize(sub, "post", "offensiveYN_n",
                  dataset_id=f"sbic_hf_offensive_{plat}", source="sbic_hf",
                  slice_desc=f"dataSource~={plat}", task="offensive_binary",
                  language="en", platform=plat,
                  provenance=f"{prov_base} Filtered to dataSource in the {plat} group; "
                             f"offensiveYN in {{0,1}} only (0.5/blank ambiguous rows dropped).")

    sub = df.loc[df["sexYN_n"].isin([0.0, 1.0]), ["post", "sexYN_n"]]
    finalize(sub, "post", "sexYN_n", dataset_id="sbic_hf_sex", source="sbic_hf",
              slice_desc="all platforms pooled", task="sexual_lewd_binary",
              language="en", platform="mixed",
              provenance=f"{prov_base} sexYN in {{0,1}} only (0.5/blank dropped); "
                         f"whether the post is sexual/lewd content.")

    sub = df.loc[df["whoTarget_n"].isin([0.0, 1.0]), ["post", "whoTarget_n"]]
    finalize(sub, "post", "whoTarget_n", dataset_id="sbic_hf_group_targeted", source="sbic_hf",
              slice_desc="all platforms pooled", task="targets_group_binary",
              language="en", platform="mixed",
              provenance=f"{prov_base} whoTarget in {{0,1}} only (blank dropped); "
                         f"whether the post targets a specific group vs an individual/none.")


def build_tweeteval():
    for cfg, name, task in [("hate", "tweeteval_hate", "hate_vs_not"),
                              ("offensive", "tweeteval_offensive", "offensive_vs_not")]:
        try:
            ds = _hf_load("cardiffnlp/tweet_eval", cfg)
            df = pd.concat([ds[s].to_pandas() for s in ds.keys()], ignore_index=True)
        except Exception as e:
            log_fail(f"{name}: DROPPED - cardiffnlp/tweet_eval[{cfg}] failed: "
                      f"{type(e).__name__}: {e}")
            continue
        finalize(df, "text", "label", dataset_id=name, source="tweet_eval",
                  slice_desc=f"config={cfg}, all splits", task=task, language="en",
                  platform="twitter",
                  provenance=f"HF dataset cardiffnlp/tweet_eval config={cfg}, "
                             f"train+validation+test concatenated.")


def build_davidson():
    try:
        ds = _hf_load("tdavidson/hate_speech_offensive")
        df = ds["train"].to_pandas()
    except Exception as e:
        log_fail(f"davidson_*: DROPPED - tdavidson/hate_speech_offensive failed: "
                  f"{type(e).__name__}: {e}")
        return
    class_map = {0: "hate_speech", 1: "offensive_language", 2: "neither"}
    df["class_name"] = df["class"].map(class_map)
    prov = "HF dataset tdavidson/hate_speech_offensive (Davidson et al. 2017 Twitter dataset)."

    finalize(df, "tweet", "class_name", dataset_id="davidson_3way", source="davidson",
              slice_desc="all", task="hate_offensive_neither_3way", language="en",
              platform="twitter",
              provenance=f"{prov} hate_speech is a small minority class (~5.8%), rebalanced "
                         f"by undersampling offensive_language/neither.")

    df["bin"] = np.where(df["class"] == 0, "hate_speech", "not_hate_speech")
    finalize(df, "tweet", "bin", dataset_id="davidson_hate_vs_not", source="davidson",
              slice_desc="all, offensive+neither collapsed", task="hate_vs_not",
              language="en", platform="twitter",
              provenance=f"{prov} offensive_language+neither collapsed into not_hate_speech; "
                         f"hate_speech minority rebalanced to the 10% floor by undersampling.")


def build_hatexplain():
    try:
        ds = _hf_load("Hate-speech-CNERG/hatexplain", revision="refs/convert/parquet")
    except Exception as e:
        log_fail(f"hatexplain_*: DROPPED - Hate-speech-CNERG/hatexplain failed: "
                  f"{type(e).__name__}: {e}")
        return
    df = pd.concat([ds[s].to_pandas() for s in ds.keys()], ignore_index=True)
    label_names = ["hatespeech", "normal", "offensive"]

    def majority(labels):
        c = Counter(labels)
        top = c.most_common()
        if len(top) > 1 and top[0][1] == top[1][1]:
            return None
        return label_names[top[0][0]]

    df["text"] = df["post_tokens"].apply(lambda toks: " ".join(toks))
    df["maj"] = df["annotators"].apply(lambda a: majority(a["label"]))
    df = df.dropna(subset=["maj"])
    prov = ("HF dataset Hate-speech-CNERG/hatexplain (loaded via "
            "revision=refs/convert/parquet), train+validation+test concatenated; "
            "text reconstructed by whitespace-joining `post_tokens` (lossy detokenization, "
            "no punctuation-spacing repair); label = majority vote across annotators "
            "(ties with no clear majority dropped); platform mixed gab+twitter (from id suffix).")

    finalize(df, "text", "maj", dataset_id="hatexplain_3way", source="hatexplain",
              slice_desc="all, majority-voted label", task="hate_normal_offensive_3way",
              language="en", platform="mixed", provenance=prov)

    df["bin"] = np.where(df["maj"] == "hatespeech", "hatespeech", "not_hatespeech")
    finalize(df, "text", "bin", dataset_id="hatexplain_hate_vs_rest", source="hatexplain",
              slice_desc="all, normal+offensive collapsed", task="hate_vs_rest",
              language="en", platform="mixed", provenance=prov)


def build_hatecheck():
    try:
        ds = _hf_load("Paul/hatecheck")
        df = ds["test"].to_pandas()
    except Exception as e:
        log_fail(f"hatecheck_en: DROPPED - Paul/hatecheck failed: {type(e).__name__}: {e}")
        return
    finalize(df, "test_case", "label_gold", dataset_id="hatecheck_en", source="hatecheck",
              slice_desc="all functional test cases", task="hateful_vs_non_hateful",
              language="en", platform="unknown",
              provenance="HF dataset Paul/hatecheck (Rottger et al. HateCheck functional "
                         "test suite; templated/synthetic sentences, not organic social-media "
                         "text -- platform is 'unknown' by design).")


def build_ucb_measuring_hate_speech():
    try:
        ds = _hf_load("ucberkeley-dlab/measuring-hate-speech", "default")
        df = ds["train"].to_pandas()
    except Exception as e:
        log_fail(f"ucb_*: DROPPED - ucberkeley-dlab/measuring-hate-speech failed: "
                  f"{type(e).__name__}: {e}")
        return

    prov_base = ("HF dataset ucberkeley-dlab/measuring-hate-speech (Kennedy et al. 2020); "
                 "row-per-annotation records aggregated to comment_id level. "
                 "platform column is a numeric code without a published label mapping in "
                 "this snapshot's features, so it is reported as 'mixed' rather than guessed.")

    g = df.groupby("comment_id").agg(
        text=("text", "first"), hs_mean=("hatespeech", "mean")
    ).reset_index()
    g["bin"] = np.where(g["hs_mean"] >= 1.0, "hate", np.where(g["hs_mean"] <= 1 / 3, "not_hate", None))
    g_bin = g.dropna(subset=["bin"])
    finalize(g_bin, "text", "bin", dataset_id="ucb_hate_binary", source="measuring_hate_speech",
              slice_desc="comment-level majority (hatespeech ordinal mean thresholded)",
              task="hate_vs_not", language="en", platform="mixed",
              provenance=f"{prov_base} hatespeech in {{0,1,2}} per annotator averaged per "
                         f"comment; mean>=1.0 -> hate, mean<=1/3 -> not_hate, else dropped "
                         f"(ambiguous).")

    target_cols = ["target_race", "target_religion", "target_origin", "target_gender",
                   "target_sexuality"]
    gt = df.groupby("comment_id").agg(
        text=("text", "first"),
        **{c: (c, "mean") for c in target_cols}
    ).reset_index()
    flags = gt[target_cols] >= 0.5
    n_flags = flags.sum(axis=1)
    single = gt[n_flags == 1].copy()
    single["target"] = flags.loc[single.index, target_cols].idxmax(axis=1).str.replace("target_", "")
    finalize(single, "text", "target", dataset_id="ucb_target_identity_5way",
              source="measuring_hate_speech",
              slice_desc="comments with exactly one majority-flagged target category "
                         "(race/religion/origin/gender/sexuality; age/disability/politics "
                         "dropped, too rare)",
              task="target_identity_5way", language="en", platform="mixed",
              provenance=f"{prov_base} target_* boolean columns averaged per comment "
                         f"(majority >= 0.5); restricted to comments with exactly one "
                         f"majority-flagged identity category among "
                         f"race/religion/origin/gender/sexuality.")


def build_ethos():
    try:
        import glob
        hits = glob.glob(os.path.expanduser(
            "~/.cache/huggingface/hub/datasets--iamollas--ethos/snapshots/*"))
        base = hits[0]
        b = pd.read_parquet(os.path.join(base, "binary", "train", "0000.parquet"))
        m = pd.read_parquet(os.path.join(base, "multilabel", "train", "0000.parquet"))
    except Exception as e:
        log_fail(f"ethos_*: DROPPED - iamollas/ethos failed: {type(e).__name__}: {e}")
        return

    prov = ("HF dataset iamollas/ethos (loaded directly from the cached "
            "refs/convert/parquet files: the `datasets` builder's own dataset_info "
            "schema mismatches the actual parquet columns for this repo, so pandas reads "
            "the parquet files directly, bypassing the broken schema cast). Comments "
            "sourced from YouTube and Reddit (paper); platform reported as 'mixed'.")

    finalize(b, "text", "label", dataset_id="ethos_binary", source="ethos",
              slice_desc="all (binary config)", task="hate_vs_not", language="en",
              platform="mixed", provenance=prov)

    cats = ["race", "religion", "gender", "national_origin"]
    for cat in cats:
        mm = m.copy()
        mm["ovr"] = np.where(mm[cat] > 0, cat, "rest")
        finalize(mm, "text", "ovr", dataset_id=f"ethos_category_{cat}_vs_rest", source="ethos",
                  slice_desc=f"multilabel config, {cat} vs rest",
                  task=f"{cat}_vs_rest_among_hate", language="en", platform="mixed",
                  provenance=f"{prov} Restricted to the 433-row multilabel subset (all "
                             f"already-hateful texts); one-vs-rest over the '{cat}' "
                             f"hate-category flag (classifies WHICH kind of hate is present, "
                             f"not hate-vs-not).")


def build_mlma():
    try:
        ds = _hf_load("nedjmaou/MLMA_hate_speech")
        df = ds["train"].to_pandas()
    except Exception as e:
        log_fail(f"mlma_*: DROPPED - nedjmaou/MLMA_hate_speech failed: {type(e).__name__}: {e}")
        return

    def script(t):
        t = str(t)
        if re.search(r"[؀-ۿ]", t):
            return "ar"
        if re.search(r"[éèàçùâêîôûëïüœ]", t.lower()):
            return "fr"
        if re.search(r"[a-zA-Z]", t):
            return "en"
        return "other"

    df["lang"] = df["tweet"].apply(script)

    def label(s):
        if s == "normal":
            return "normal"
        if "hateful" in s or "abusive" in s:
            return "hateful_or_abusive"
        return None

    df["lab"] = df["sentiment"].apply(label)
    prov = ("HF dataset nedjmaou/MLMA_hate_speech (Ousidhoum et al. 2019, multilingual "
            "en/fr/ar hate-speech tweets). No explicit language column in this snapshot; "
            "language assigned heuristically per tweet (Arabic-script regex -> ar, French "
            "accented-character regex -> fr, else Latin-script -> en) -- a documented "
            "heuristic with some en/fr crosstalk, not ground truth. Label collapsed from "
            "the raw multi-aspect `sentiment` field: rows containing 'hateful' or 'abusive' "
            "-> hateful_or_abusive, rows == 'normal' -> normal; purely 'offensive'/"
            "'disrespectful' rows (ambiguous middle ground) dropped.")

    for lang in ["en", "fr", "ar"]:
        sub = df[(df["lang"] == lang) & df["lab"].notna()]
        finalize(sub, "tweet", "lab", dataset_id=f"mlma_{lang}_hate_vs_normal", source="mlma",
                  slice_desc=f"lang={lang} (heuristic)", task="hateful_or_abusive_vs_normal",
                  language=lang, platform="twitter", provenance=prov)


def build_hatebr():
    try:
        ds = _hf_load("ruanchaves/hatebr", revision="refs/convert/parquet")
        df = pd.concat([ds[s].to_pandas() for s in ds.keys()], ignore_index=True)
    except Exception as e:
        log_fail(f"hatebr_*: DROPPED - ruanchaves/hatebr failed: {type(e).__name__}: {e}")
        return
    prov = ("HF dataset ruanchaves/hatebr (loaded via revision=refs/convert/parquet), "
            "Portuguese Instagram comments (Vargas et al. 2022); platform 'unknown' since "
            "Instagram is not one of the enumerated platform values.")

    finalize(df, "instagram_comments", "offensive_language", dataset_id="hatebr_offensive_binary",
              source="hatebr", slice_desc="all splits", task="offensive_binary",
              language="pt", platform="unknown", provenance=prov)

    df["ovr"] = np.where(df["partyism"], "partyism", "not_partyism")
    finalize(df, "instagram_comments", "ovr", dataset_id="hatebr_partyism_vs_rest",
              source="hatebr", slice_desc="all splits, partyism sub-category vs rest",
              task="partyism_vs_rest", language="pt", platform="unknown",
              provenance=f"{prov} 'partyism' (political-affiliation-based hate) is a minority "
                         f"sub-category (~7.5%), rebalanced by undersampling the rest.")


def build_toxic_conversations():
    try:
        ds = _hf_load("SetFit/toxic_conversations")
        df = ds["train"].to_pandas()
    except Exception as e:
        log_fail(f"toxic_conversations_binary: DROPPED - SetFit/toxic_conversations failed: "
                  f"{type(e).__name__}: {e}")
        return
    if len(df) > 60000:
        df = df.sample(n=60000, random_state=SEED)
    finalize(df, "text", "label", dataset_id="toxic_conversations_binary",
              source="toxic_conversations", slice_desc="random 60k subsample of train split",
              task="toxic_binary", language="en", platform="unknown",
              provenance="HF dataset SetFit/toxic_conversations (Jigsaw Unintended Bias in "
                         "Toxicity Classification / Civil Comments); toxic is a ~8% minority, "
                         "rebalanced by undersampling non-toxic; platform 'unknown' (web "
                         "comments, not twitter/reddit/gab/stormfront).")


LOCAL_BUILDERS = [
    ("IHC (Implicit Hate Corpus)", build_ihc),
    ("HatEval (SemEval-2019 Task 5)", build_hateval),
    ("SBIC (local terms_clean cut)", build_sbic_local),
    ("HateWiC", build_hatewic),
]

HF_BUILDERS = [
    ("SBIC (HF allenai/social_bias_frames)", build_sbic_hf),
    ("tweet_eval (hate, offensive)", build_tweeteval),
    ("Davidson hate_speech_offensive", build_davidson),
    ("HateXplain", build_hatexplain),
    ("HateCheck", build_hatecheck),
    ("Measuring Hate Speech (UC Berkeley)", build_ucb_measuring_hate_speech),
    ("ETHOS", build_ethos),
    ("MLMA_hate_speech (multilingual)", build_mlma),
    ("HateBR (Portuguese)", build_hatebr),
    ("toxic_conversations (Jigsaw/Civil Comments)", build_toxic_conversations),
]


def run_all():
    os.makedirs(DATASETS_DIR, exist_ok=True)

    log("# Local sources\n")
    for name, fn in LOCAL_BUILDERS:
        log(f"\n## {name}")
        try:
            fn()
        except Exception as e:
            log_fail(f"{name}: top-level FAILURE - {type(e).__name__}: {e}")
            log(traceback.format_exc())

    log("\n# HuggingFace sources (bonus)\n")
    for name, fn in HF_BUILDERS:
        log(f"\n## {name}")
        try:
            fn()
        except Exception as e:
            log_fail(f"{name}: top-level FAILURE - {type(e).__name__}: {e}")
            log(traceback.format_exc())


def verify(manifest):
    problems = []
    on_disk = {f[:-len(".parquet")] for f in os.listdir(DATASETS_DIR) if f.endswith(".parquet")}
    in_manifest = set(manifest["dataset_id"])

    if on_disk != in_manifest:
        problems.append(f"parquet files vs manifest mismatch: "
                         f"only_on_disk={on_disk - in_manifest} only_in_manifest={in_manifest - on_disk}")

    for _, row in manifest.iterrows():
        did = row["dataset_id"]
        path = os.path.join(DATASETS_DIR, f"{did}.parquet")
        if not os.path.exists(path):
            problems.append(f"{did}: parquet file missing")
            continue
        df = pd.read_parquet(path)
        if list(df.columns) != ["text", "label"]:
            problems.append(f"{did}: columns are {list(df.columns)}, expected ['text','label']")
        if df["text"].isnull().any() or (df["text"].astype(str).str.strip() == "").any():
            problems.append(f"{did}: contains null/empty text")
        if df["text"].duplicated().any():
            problems.append(f"{did}: contains duplicate text rows")
        if not np.issubdtype(df["label"].dtype, np.integer):
            problems.append(f"{did}: label dtype is {df['label'].dtype}, expected int")
        labs = sorted(df["label"].unique())
        if labs != list(range(len(labs))):
            problems.append(f"{did}: labels are not a contiguous 0..k-1 range: {labs}")
        n = len(df)
        if n > CAP:
            problems.append(f"{did}: n_instances={n} exceeds cap {CAP}")
        if n < MIN_TOTAL:
            problems.append(f"{did}: n_instances={n} below minimum {MIN_TOTAL}")
        counts = df["label"].value_counts()
        if len(counts) < 2:
            problems.append(f"{did}: fewer than 2 classes")
        if (counts < MIN_CLASS_N).any():
            problems.append(f"{did}: a class has < {MIN_CLASS_N} instances: {counts.to_dict()}")
        min_frac = counts.min() / n
        if min_frac < MIN_FRAC - 1e-6:
            problems.append(f"{did}: minority frac {min_frac:.4f} < {MIN_FRAC}")
        if row["n_instances"] != n:
            problems.append(f"{did}: manifest n_instances={row['n_instances']} != actual {n}")
        if row["n_classes"] != len(counts):
            problems.append(f"{did}: manifest n_classes={row['n_classes']} != actual {len(counts)}")

    required_cols = ["dataset_id", "source", "slice", "task", "language", "platform",
                      "n_instances", "n_classes", "minority_pct", "provenance"]
    if list(manifest.columns) != required_cols:
        problems.append(f"manifest columns are {list(manifest.columns)}, expected {required_cols}")

    return problems


def write_manifest_and_notes(runtime_s):
    manifest = pd.DataFrame(MANIFEST_ROWS).sort_values("dataset_id").reset_index(drop=True)
    manifest.to_csv(MANIFEST_PATH, index=False)

    problems = verify(manifest)

    with open(NOTES_PATH, "w") as f:
        f.write("# datasets_notes.md\n\n")
        f.write("Provenance and build log for `data_hs/datasets/` "
                "(hate-speech meta-dataset acquisition layer).\n\n")
        f.write(f"Built by `build_datasets.py`. Runtime: {runtime_s:.1f}s. "
                f"Seed: {SEED}. Cap: {CAP}. Min total: {MIN_TOTAL}. "
                f"Min minority frac: {MIN_FRAC}. Min class n: {MIN_CLASS_N}.\n\n")
        f.write(f"## Result: {len(manifest)} datasets written\n\n")

        f.write("### Summary table\n\n")
        f.write(manifest[["dataset_id", "source", "task", "language", "n_instances",
                           "n_classes", "minority_pct"]].to_markdown(index=False))
        f.write("\n\n")

        f.write("### What FAILED / was dropped, and why\n\n")
        if FAIL_NOTES:
            for line in FAIL_NOTES:
                f.write(f"- {line}\n")
        else:
            f.write("(nothing failed)\n")
        f.write("\n")

        f.write("### Diversity notes\n\n")
        f.write(f"- Languages: {sorted(manifest['language'].unique().tolist())}\n")
        f.write(f"- Platforms: {sorted(manifest['platform'].unique().tolist())}\n")
        f.write(f"- Sources: {sorted(manifest['source'].unique().tolist())}\n")
        f.write(f"- n_classes distribution: {manifest['n_classes'].value_counts().to_dict()}\n")
        f.write("- Deliberately spans: explicit vs implicit hate (IHC), offensive vs hate "
                "(Davidson/HateXplain/tweet_eval), target/identity identification (IHC stg3 "
                "groups, UC Berkeley target_identity_5way, SBIC whoTarget), word-in-context "
                "hate sense (HateWiC), synthetic functional testing (HateCheck), and varying "
                "imbalance levels (from near-50/50 to the 10% rebalanced floor).\n\n")

        f.write("### Verification against hard rules\n\n")
        if problems:
            f.write(f"**{len(problems)} PROBLEMS FOUND:**\n\n")
            for p in problems:
                f.write(f"- {p}\n")
        else:
            f.write("All datasets pass every hard rule (dedup, no null text, "
                    ">=2 classes >=30 instances, minority >=10%, "
                    f"{MIN_TOTAL} <= n <= {CAP}, manifest counts match parquet).\n")
        f.write("\n")

        f.write("### Full build log\n\n```\n")
        f.write("\n".join(NOTES))
        f.write("\n```\n")

    return manifest, problems


if __name__ == "__main__":
    t0 = time.time()
    run_all()
    manifest, problems = write_manifest_and_notes(time.time() - t0)
    print("\n" + "=" * 70)
    print(f"Wrote {len(manifest)} datasets to {DATASETS_DIR}")
    print(f"Manifest: {MANIFEST_PATH}")
    print(f"Notes: {NOTES_PATH}")
    if problems:
        print(f"\n{len(problems)} VERIFICATION PROBLEMS:")
        for p in problems:
            print(f" - {p}")
        sys.exit(1)
    else:
        print("All hard-rule assertions passed.")
