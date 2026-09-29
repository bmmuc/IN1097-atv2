import os
import re
import urllib.request
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.neighbors import NearestCentroid, NearestNeighbors
from sklearn.tree import DecisionTreeClassifier

RANDOM_STATE = 42

# tok_ -- tokenizer fertility (Rust et al. 2021, "How good is your tokenizer?")
TOKENIZER_MODEL_IDS = {
    "hatebert": "GroNLP/hateBERT",
    "mbert": "google-bert/bert-base-multilingual-cased",
}

# module-level lazy cache: one tokenizer per tag per worker process
_TOKENIZER_CACHE = {}


def _get_tokenizer(tag):
    if tag not in _TOKENIZER_CACHE:
        from transformers import AutoTokenizer

        model_id = TOKENIZER_MODEL_IDS[tag]
        try:
            tok = AutoTokenizer.from_pretrained(model_id, local_files_only=True)
        except Exception:
            tok = AutoTokenizer.from_pretrained(model_id)
        _TOKENIZER_CACHE[tag] = tok
    return _TOKENIZER_CACHE[tag]


def compute_tok_features(texts):
    """Subword-fertility descriptors of the hateBERT/mBERT tokenizers on the dataset words."""
    words_per_doc = [t.split() for t in texts]
    word_counter = Counter()
    for ws in words_per_doc:
        word_counter.update(ws)
    unique_words = list(word_counter.keys())
    counts = np.array([word_counter[w] for w in unique_words], dtype=float)
    total = float(counts.sum()) if len(counts) else 0.0

    feats = {}
    fertility_by_tag = {}
    for tag in ("hatebert", "mbert"):
        if total == 0:
            fertility_by_tag[tag] = 0.0
            feats[f"tok_fertility_{tag}"] = 0.0
            feats[f"tok_continued_{tag}"] = 0.0
            feats[f"tok_unk_{tag}"] = 0.0
            continue

        tok = _get_tokenizer(tag)
        unk_id = tok.unk_token_id
        enc = tok(unique_words, add_special_tokens=False)
        ids_list = enc["input_ids"]
        n_pieces = np.array([max(len(ids), 1) for ids in ids_list], dtype=float)
        has_unk = np.array(
            [unk_id is not None and unk_id in ids for ids in ids_list], dtype=bool
        )

        fertility = float(np.sum(counts * n_pieces) / total)
        continued = float(np.sum(counts[n_pieces >= 2]) / total)
        unk_rate = float(np.sum(counts[has_unk]) / total)

        fertility_by_tag[tag] = fertility
        feats[f"tok_fertility_{tag}"] = fertility
        feats[f"tok_continued_{tag}"] = continued
        feats[f"tok_unk_{tag}"] = unk_rate

    feats["tok_fertility_gap"] = fertility_by_tag["hatebert"] - fertility_by_tag["mbert"]
    return feats


# hl_ -- HurtLex lexicon coverage (Bassignana, Basile & Patti 2018; Wiegand et al. 2019)
HURTLEX_LANGS = {"en": "EN", "es": "ES", "pt": "PT", "fr": "FR", "ar": "AR"}
HURTLEX_URL = "https://raw.githubusercontent.com/valeriobasile/hurtlex/master/lexica/{tag}/1.2/hurtlex_{tag}.tsv"

_LEXICON_CACHE = {}


def ensure_hurtlex_lexicon(language, lexica_dir):
    """Download HurtLex v1.2 for `language` if missing; returns the path, or None if unsupported."""
    tag = HURTLEX_LANGS.get((language or "").strip().lower())
    if tag is None:
        return None
    os.makedirs(lexica_dir, exist_ok=True)
    path = os.path.join(lexica_dir, f"hurtlex_{tag}.tsv")
    if not os.path.exists(path):
        url = HURTLEX_URL.format(tag=tag)
        tmp_path = path + ".tmp"
        urllib.request.urlretrieve(url, tmp_path)
        os.replace(tmp_path, path)
    return path


def load_hurtlex_lexicon(path):
    """Parse a HurtLex TSV into single-word and multi-word (n-gram tuple) lemma -> categories maps."""
    if path in _LEXICON_CACHE:
        return _LEXICON_CACHE[path]
    df = pd.read_csv(path, sep="\t", dtype=str)
    single = defaultdict(list)
    multi_index = defaultdict(list)
    max_len = 0
    for lemma, cat in zip(df["lemma"].astype(str), df["category"].astype(str)):
        lemma_l = lemma.strip().lower()
        if not lemma_l:
            continue
        words = tuple(lemma_l.split())
        if len(words) == 1:
            single[words[0]].append(cat)
        else:
            multi_index[words].append(cat)
            max_len = max(max_len, len(words))
    lex = {"single": dict(single), "multi_index": dict(multi_index), "max_len": max_len}
    _LEXICON_CACHE[path] = lex
    return lex


def get_lexicon_for_language(language, lexica_dir):
    path = ensure_hurtlex_lexicon(language, lexica_dir)
    if path is None:
        return None
    return load_hurtlex_lexicon(path)


def compute_hl_features(texts, labels, lexicon):
    """HurtLex doc/token coverage, class gap and category diversity (multi-word lemmas matched as n-grams)."""
    labels = np.asarray(labels)
    n = len(texts)
    if lexicon is None:
        return {"hl_doc_rate": 0.0, "hl_token_rate": 0.0, "hl_class_gap": 0.0, "hl_category_entropy": 0.0}

    single = lexicon["single"]
    multi_index = lexicon["multi_index"]
    max_len = lexicon["max_len"]

    doc_has_hit = np.zeros(n, dtype=bool)
    single_hits = 0
    total_tokens = 0
    category_counter = Counter()

    for i, text in enumerate(texts):
        tokens = re.findall(r"\w+", text.lower())
        total_tokens += len(tokens)
        hit = False
        for tok in tokens:
            cats = single.get(tok)
            if cats:
                hit = True
                single_hits += 1
                category_counter.update(cats)
        m = len(tokens)
        if max_len >= 2:
            for start in range(m):
                for length in range(2, max_len + 1):
                    if start + length > m:
                        break
                    cats = multi_index.get(tuple(tokens[start:start + length]))
                    if cats:
                        hit = True
                        category_counter.update(cats)
        doc_has_hit[i] = hit

    hl_doc_rate = float(np.mean(doc_has_hit)) if n else 0.0
    hl_token_rate = single_hits / total_tokens if total_tokens else 0.0

    classes = np.unique(labels)
    class_rates = [float(np.mean(doc_has_hit[labels == c])) for c in classes]
    hl_class_gap = float(max(class_rates) - min(class_rates)) if class_rates else 0.0

    if category_counter:
        total_cat = sum(category_counter.values())
        probs = np.array([v / total_cat for v in category_counter.values()])
        hl_category_entropy = float(-np.sum(probs * np.log2(probs)))
    else:
        hl_category_entropy = 0.0

    return {
        "hl_doc_rate": hl_doc_rate,
        "hl_token_rate": hl_token_rate,
        "hl_class_gap": hl_class_gap,
        "hl_category_entropy": hl_category_entropy,
    }


# ch_ -- corpus hardness (Pinto & Rosso 2007; Madrid, Escalante & Morales 2019)
def _js_divergence(p, q):
    """Jensen-Shannon divergence, base 2, in [0, 1]."""
    m = 0.5 * (p + q)

    def kl(a, b):
        mask = a > 0
        if not np.any(mask):
            return 0.0
        return float(np.sum(a[mask] * np.log2(a[mask] / b[mask])))

    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


def compute_ch_features(texts, labels):
    labels = np.asarray(labels)
    doc_tokens = [re.findall(r"\w+", t.lower()) for t in texts]

    corpus_counter = Counter()
    for toks in doc_tokens:
        corpus_counter.update(toks)

    vocab = [w for w, c in corpus_counter.items() if c >= 2]
    vocab_index = {w: i for i, w in enumerate(vocab)}
    V = len(vocab)

    classes = np.unique(labels)
    class_counters = defaultdict(Counter)
    for toks, lab in zip(doc_tokens, labels):
        class_counters[lab].update(toks)

    class_dists = {}
    class_vocabs = {}
    for c in classes:
        counter_c = class_counters[c]
        arr = np.zeros(V)
        for w, cnt in counter_c.items():
            j = vocab_index.get(w)
            if j is not None:
                arr[j] = cnt
        total_c = arr.sum()
        class_dists[c] = arr / total_c if total_c > 0 else arr
        class_vocabs[c] = {w for w, cnt in counter_c.items() if cnt >= 2}

    corpus_arr = np.array([corpus_counter[w] for w in vocab], dtype=float)
    corpus_dist = corpus_arr / corpus_arr.sum() if corpus_arr.sum() > 0 else corpus_arr

    pair_js = []
    jaccards = []
    class_list = list(classes)
    for i in range(len(class_list)):
        for j in range(i + 1, len(class_list)):
            ci, cj = class_list[i], class_list[j]
            pair_js.append(_js_divergence(class_dists[ci], class_dists[cj]))
            vi, vj = class_vocabs[ci], class_vocabs[cj]
            union = vi | vj
            jaccards.append(len(vi & vj) / len(union) if union else 0.0)

    ch_js_between_classes = float(np.mean(pair_js)) if pair_js else 0.0
    ch_vocab_jaccard = float(np.mean(jaccards)) if jaccards else 0.0

    js_to_corpus = [_js_divergence(class_dists[c], corpus_dist) for c in classes]
    ch_js_class_to_corpus = float(np.mean(js_to_corpus)) if js_to_corpus else 0.0

    top_terms = corpus_counter.most_common(1000)
    freqs = np.array([c for _, c in top_terms], dtype=float)
    ranks = np.arange(1, len(freqs) + 1, dtype=float)
    if len(freqs) >= 2:
        ch_zipf_slope = float(np.polyfit(np.log10(ranks), np.log10(freqs), 1)[0])
    else:
        ch_zipf_slope = 0.0

    return {
        "ch_js_between_classes": ch_js_between_classes,
        "ch_js_class_to_corpus": ch_js_class_to_corpus,
        "ch_vocab_jaccard": ch_vocab_jaccard,
        "ch_zipf_slope": ch_zipf_slope,
    }


# lm_ -- landmarkers (Pfahringer, Bensusan & Giraud-Carrier 2000; Furnkranz & Petrak 2001)
def _l2_normalize(emb):
    norms = np.maximum(np.linalg.norm(emb, axis=1, keepdims=True), 1e-12)
    return emb / norms


def compute_lm_features(
    texts,
    labels,
    minilm_emb,
    hatebert_emb,
    random_state=RANDOM_STATE,
):
    labels = np.asarray(labels)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
    folds = list(skf.split(texts, labels))

    def cv_text_f1(make_pipeline_fn):
        scores = []
        for train_idx, test_idx in folds:
            X_train = [texts[i] for i in train_idx]
            X_test = [texts[i] for i in test_idx]
            vec, clf = make_pipeline_fn()
            Xt = vec.fit_transform(X_train)
            clf.fit(Xt, labels[train_idx])
            preds = clf.predict(vec.transform(X_test))
            scores.append(f1_score(labels[test_idx], preds, average="macro"))
        return float(np.mean(scores))

    def cv_array_f1(X):
        scores = []
        for train_idx, test_idx in folds:
            clf = NearestCentroid()
            clf.fit(X[train_idx], labels[train_idx])
            preds = clf.predict(X[test_idx])
            scores.append(f1_score(labels[test_idx], preds, average="macro"))
        return float(np.mean(scores))

    f1_stump = cv_text_f1(lambda: (
        CountVectorizer(binary=True, min_df=2, max_features=5000),
        DecisionTreeClassifier(max_depth=1, class_weight="balanced", random_state=random_state),
    ))
    f1_tree3 = cv_text_f1(lambda: (
        CountVectorizer(binary=True, min_df=2, max_features=5000),
        DecisionTreeClassifier(max_depth=3, class_weight="balanced", random_state=random_state),
    ))
    f1_nc_tfidf = cv_text_f1(lambda: (
        TfidfVectorizer(sublinear_tf=True, min_df=2),
        NearestCentroid(),
    ))

    f1_nc_minilm = cv_array_f1(_l2_normalize(minilm_emb))
    f1_nc_hatebert = cv_array_f1(_l2_normalize(hatebert_emb))

    return {
        "lm_stump_bow_f1": f1_stump,
        "lm_tree3_bow_f1": f1_tree3,
        "lm_nc_tfidf_f1": f1_nc_tfidf,
        "lm_nc_minilm_f1": f1_nc_minilm,
        "lm_nc_hatebert_f1": f1_nc_hatebert,
        "lm_nc_gap_minilm_minus_tfidf": f1_nc_minilm - f1_nc_tfidf,
    }


# cx_ -- complexity (Ho & Basu 2002; Lorena et al. 2019)
def compute_cx_features(minilm_emb, labels, random_state=RANDOM_STATE):
    """Complexity measures (Ho & Basu 2002; Lorena et al. 2019) in a PCA(<=50) space of MiniLM embeddings."""
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import minimum_spanning_tree
    from scipy.spatial.distance import pdist, squareform
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    labels = np.asarray(labels)
    n, d = minilm_emb.shape
    classes = np.unique(labels)

    scaler = StandardScaler()
    Zs = scaler.fit_transform(minilm_emb)
    n_comp = max(1, min(50, n - 1))
    pca = PCA(n_components=n_comp, random_state=random_state)
    Z = pca.fit_transform(Zs)

    D = squareform(pdist(Z, metric="euclidean"))

    # --- F1: max Fisher discriminant ratio (normalized, lower = simpler) ---
    mu = Z.mean(axis=0)
    num = np.zeros(Z.shape[1])
    den = np.zeros(Z.shape[1])
    for c in classes:
        Zc = Z[labels == c]
        mu_c = Zc.mean(axis=0)
        num += Zc.shape[0] * (mu_c - mu) ** 2
        den += ((Zc - mu_c) ** 2).sum(axis=0)
    ratio = np.where(den > 1e-12, num / np.where(den > 1e-12, den, 1.0), np.where(num > 1e-12, 1e12, 0.0))
    f1_max = float(np.max(ratio)) if len(ratio) else 0.0
    cx_f1 = 1.0 / (1.0 + f1_max)

    # --- N1: fraction of points incident to an inter-class MST edge ---
    mst = minimum_spanning_tree(csr_matrix(D)).tocoo()
    diff_class_nodes = set()
    for i, j in zip(mst.row, mst.col):
        if labels[i] != labels[j]:
            diff_class_nodes.add(int(i))
            diff_class_nodes.add(int(j))
    cx_n1 = len(diff_class_nodes) / n

    # --- N2 and LSC share the same masked-distance setup ---
    same_mask = labels[:, None] == labels[None, :]
    Dq = D.copy()
    np.fill_diagonal(Dq, np.inf)
    D_same = np.where(same_mask, Dq, np.inf)
    D_diff = np.where(~same_mask, Dq, np.inf)
    nearest_same = np.where(np.isinf(D_same.min(axis=1)), 0.0, D_same.min(axis=1))
    nearest_diff = D_diff.min(axis=1)
    r = float(nearest_same.sum() / max(float(nearest_diff.sum()), 1e-12))
    cx_n2 = r / (1.0 + r)

    ls_counts = np.zeros(n)
    row_max = D.max(axis=1)
    for i in range(n):
        enemy_d = D_diff[i].min()
        if np.isinf(enemy_d):
            enemy_d = row_max[i] + 1.0
        ls_counts[i] = np.sum(D[i] < enemy_d)
    cx_lsc = float(1.0 - ls_counts.sum() / (n ** 2))

    # --- T2, T3, T4 (cheap, on the original/standardized embedding) ---
    n_comp95 = max(1, min(d, n - 1))
    pca_full = PCA(n_components=n_comp95, random_state=random_state)
    pca_full.fit(Zs)
    cumvar = np.cumsum(pca_full.explained_variance_ratio_)
    n95 = int(np.searchsorted(cumvar, 0.95) + 1)
    n95 = min(n95, n_comp95)
    cx_t2 = d / n
    cx_t3 = n95 / n
    cx_t4 = n95 / d

    # network measures (Garcia, de Carvalho & Lorena 2015); eps is the 0.15 quantile of pairwise
    # distances, since a fixed 0.15 * max distance gives a degenerate graph in this dense space
    iu = np.triu_indices(n, k=1)
    eps = float(np.quantile(D[iu], 0.15)) if len(iu[0]) else 0.0
    adj = (D <= eps) & (~np.eye(n, dtype=bool)) & same_mask
    n_edges = int(adj.sum() // 2)
    cx_density = 1.0 - (2.0 * n_edges) / (n * (n - 1))

    # cx_hubs follows ECoL: hub scores scaled by their max; 1.0 when there are no edges
    try:
        import networkx as nx

        G = nx.from_numpy_array(adj.astype(int))
        clustering = nx.average_clustering(G) if n_edges > 0 else 0.0
        cx_clscoef = 1.0 - float(clustering)
        if n_edges > 0:
            try:
                hubs, _ = nx.hits(G, max_iter=1000, normalized=True)
                hub_vals = np.array(list(hubs.values()), dtype=float)
                max_hub = hub_vals.max() if len(hub_vals) else 0.0
                scaled_hub = hub_vals / max_hub if max_hub > 0 else hub_vals
                mean_scaled_hub = float(np.mean(scaled_hub))
            except Exception:
                mean_scaled_hub = 0.0
            cx_hubs = 1.0 - mean_scaled_hub
        else:
            cx_hubs = 1.0
    except ImportError:
        A = adj.astype(float)
        deg = A.sum(axis=1)
        A3 = A @ A @ A
        triangles = np.diag(A3)
        denom = deg * (deg - 1)
        local_cc = np.where(denom > 0, triangles / np.where(denom > 0, denom, 1.0), 0.0)
        cx_clscoef = 1.0 - float(np.mean(local_cc))
        if n_edges > 0:
            h = np.ones(n)
            for _ in range(50):
                a = A.T @ h
                a_norm = np.linalg.norm(a)
                a = a / a_norm if a_norm > 0 else a
                h = A @ a
                h_norm = np.linalg.norm(h)
                h = h / h_norm if h_norm > 0 else h
            max_h = h.max()
            scaled_h = h / max_h if max_h > 0 else h
            cx_hubs = 1.0 - float(np.mean(scaled_h))
        else:
            cx_hubs = 1.0

    return {
        "cx_f1": cx_f1,
        "cx_n1": cx_n1,
        "cx_n2": cx_n2,
        "cx_lsc": cx_lsc,
        "cx_t2": cx_t2,
        "cx_t3": cx_t3,
        "cx_t4": cx_t4,
        "cx_density": cx_density,
        "cx_clscoef": cx_clscoef,
        "cx_hubs": cx_hubs,
    }


# shared kNN (used by ih_ and mt_) -- Smith, Martinez & Giraud-Carrier 2014
def _knn_same_class(X, labels, k=5, metric="cosine"):
    """Count, for each point, how many of its k nearest neighbours share its label (dense or sparse X)."""
    n = X.shape[0]
    k_eff = max(1, min(k, n - 1))
    nn = NearestNeighbors(n_neighbors=k_eff + 1, metric=metric, n_jobs=-1).fit(X)
    _, idx = nn.kneighbors(X)

    out = np.empty((n, k_eff), dtype=idx.dtype)
    self_col = np.arange(n)
    for i in range(n):
        row = idx[i][idx[i] != self_col[i]]
        out[i] = row[:k_eff] if len(row) >= k_eff else np.pad(row, (0, k_eff - len(row)), mode="edge")

    same = (labels[out] == labels[:, None]).sum(axis=1)
    return same, out, k_eff


def build_hardness_tfidf(texts):
    """Label-blind TF-IDF matrix shared by ih_kdn_tfidf and mt_*_tfidf."""
    vec = TfidfVectorizer(sublinear_tf=True, min_df=2)
    return vec.fit_transform(texts)


def compute_shared_knn(minilm_emb, tfidf_matrix, labels, k=5):
    labels = np.asarray(labels)
    same_minilm, idx_minilm, k_minilm = _knn_same_class(minilm_emb, labels, k=k, metric="cosine")
    same_tfidf, idx_tfidf, k_tfidf = _knn_same_class(tfidf_matrix, labels, k=k, metric="cosine")
    return {
        "same_minilm": same_minilm, "idx_minilm": idx_minilm, "k_minilm": k_minilm,
        "same_tfidf": same_tfidf, "idx_tfidf": idx_tfidf, "k_tfidf": k_tfidf,
    }


# ih_ -- instance hardness (Smith, Martinez & Giraud-Carrier 2014)
def compute_ih_features(
    minilm_emb,
    hatebert_emb,
    mbert_emb,
    tfidf_matrix,
    labels,
    shared_knn=None,
    k=5,
):
    labels = np.asarray(labels)
    if shared_knn is not None:
        same_minilm, k_minilm = shared_knn["same_minilm"], shared_knn["k_minilm"]
        same_tfidf, k_tfidf = shared_knn["same_tfidf"], shared_knn["k_tfidf"]
    else:
        same_minilm, _, k_minilm = _knn_same_class(minilm_emb, labels, k=k, metric="cosine")
        same_tfidf, _, k_tfidf = _knn_same_class(tfidf_matrix, labels, k=k, metric="cosine")

    same_hatebert, _, k_hatebert = _knn_same_class(hatebert_emb, labels, k=k, metric="cosine")
    same_mbert, _, k_mbert = _knn_same_class(mbert_emb, labels, k=k, metric="cosine")

    kdn_minilm = (k_minilm - same_minilm) / k_minilm
    kdn_hatebert = (k_hatebert - same_hatebert) / k_hatebert
    kdn_mbert = (k_mbert - same_mbert) / k_mbert
    kdn_tfidf = (k_tfidf - same_tfidf) / k_tfidf

    return {
        "ih_kdn_minilm": float(np.mean(kdn_minilm)),
        "ih_kdn_hatebert": float(np.mean(kdn_hatebert)),
        "ih_kdn_mbert": float(np.mean(kdn_mbert)),
        "ih_kdn_tfidf": float(np.mean(kdn_tfidf)),
        "ih_hard_frac_minilm": float(np.mean(kdn_minilm > 0.5)),
        "ih_kdn_gap_tfidf_minus_minilm": float(np.mean(kdn_tfidf) - np.mean(kdn_minilm)),
    }


# mt_ -- minority-class typology (Napierala & Stefanowski 2016)
def compute_mt_features(labels, shared_knn):
    """Minority-class typology buckets; assumes shared_knn was built with k=5."""
    labels = np.asarray(labels)
    counts = Counter(labels.tolist())
    minority_label = min(counts, key=lambda c: counts[c])
    minority_mask = labels == minority_label

    def typology(same_counts):
        s = same_counts[minority_mask]
        if len(s) == 0:
            return 0.0, 0.0, 0.0, 0.0
        safe = float(np.mean(np.isin(s, [4, 5])))
        borderline = float(np.mean(np.isin(s, [2, 3])))
        rare = float(np.mean(s == 1))
        outlier = float(np.mean(s == 0))
        return safe, borderline, rare, outlier

    safe_m, border_m, rare_m, outlier_m = typology(shared_knn["same_minilm"])
    safe_t, _, _, outlier_t = typology(shared_knn["same_tfidf"])

    return {
        "mt_safe": safe_m,
        "mt_borderline": border_m,
        "mt_rare": rare_m,
        "mt_outlier": outlier_m,
        "mt_safe_tfidf": safe_t,
        "mt_outlier_tfidf": outlier_t,
    }


# lc_ -- learning curve (Leite & Brazdil 2005)
def _stratified_subsample_idx(idx_pool, labels_pool, size, rng):
    """Stratified subsample of `size` indices from idx_pool with at least one instance per class."""
    classes, counts = np.unique(labels_pool, return_counts=True)
    size = min(size, len(idx_pool))
    props = counts / counts.sum()
    alloc = np.maximum(1, np.floor(props * size).astype(int))
    alloc = np.minimum(alloc, counts)

    diff = size - alloc.sum()
    order = np.argsort(-counts)
    guard = 0
    while diff != 0 and guard < 10000:
        c = order[guard % len(order)]
        if diff > 0 and alloc[c] < counts[c]:
            alloc[c] += 1
            diff -= 1
        elif diff < 0 and alloc[c] > 1:
            alloc[c] -= 1
            diff += 1
        guard += 1

    chosen = []
    for c, a in zip(classes, alloc):
        cls_idx = idx_pool[labels_pool == c]
        chosen.extend(rng.choice(cls_idx, size=int(a), replace=False))
    return np.array(chosen)


def compute_lc_features(texts, labels, minilm_emb, random_state=RANDOM_STATE):
    labels = np.asarray(labels)
    n = len(labels)
    idx_all = np.arange(n)
    train_idx, test_idx = train_test_split(
        idx_all, test_size=0.3, stratify=labels, random_state=random_state
    )
    y_test = labels[test_idx]
    minilm_norm = _l2_normalize(minilm_emb)

    sizes = [s for s in (25, 50, 100, 200) if s <= len(train_idx)]
    mean_f1_minilm, mean_f1_tfidf = [], []

    for size in sizes:
        f1s_minilm, f1s_tfidf = [], []
        for seed in (0, 1, 2):
            rng = np.random.RandomState(seed)
            sub_idx = _stratified_subsample_idx(train_idx, labels[train_idx], size, rng)

            clf = NearestCentroid()
            clf.fit(minilm_norm[sub_idx], labels[sub_idx])
            preds = clf.predict(minilm_norm[test_idx])
            f1s_minilm.append(f1_score(y_test, preds, average="macro"))

            vec = TfidfVectorizer(sublinear_tf=True, min_df=1)
            X_sub = vec.fit_transform([texts[i] for i in sub_idx])
            X_test_v = vec.transform([texts[i] for i in test_idx])
            clf2 = NearestCentroid()
            clf2.fit(X_sub, labels[sub_idx])
            preds2 = clf2.predict(X_test_v)
            f1s_tfidf.append(f1_score(y_test, preds2, average="macro"))

        mean_f1_minilm.append(float(np.mean(f1s_minilm)))
        mean_f1_tfidf.append(float(np.mean(f1s_tfidf)))

    if len(sizes) >= 2:
        log2sizes = np.log2(sizes)
        slope_minilm = float(np.polyfit(log2sizes, mean_f1_minilm, 1)[0])
        slope_tfidf = float(np.polyfit(log2sizes, mean_f1_tfidf, 1)[0])
    else:
        slope_minilm = 0.0
        slope_tfidf = 0.0

    gap = float(np.mean(np.array(mean_f1_minilm) - np.array(mean_f1_tfidf))) if sizes else 0.0

    return {
        "lc_slope_minilm": slope_minilm,
        "lc_slope_tfidf": slope_tfidf,
        "lc_auc_gap_minilm_minus_tfidf": gap,
    }
