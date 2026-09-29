import re
from collections import Counter

import numpy as np
from sklearn.feature_selection import mutual_info_classif
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import cosine_similarity, cosine_distances

# regexes/lexicons for the social-media / domain-specific descriptors
MENTION_RE = re.compile(r"@\w+")
URL_RE = re.compile(r"https?://|www\.")
HASHTAG_RE = re.compile(r"#\w+")
ELONGATION_RE = re.compile(r"(.)\1{2,}")
EXCLAMATION_RE = re.compile(r"!")

# unicode emoji blocks (avoids an external dependency)
EMOJI_RE = re.compile(
    "["
    "\U0001F300-\U0001FAFF"  # misc symbols & pictographs, emoticons, transport, supplemental symbols
    "\U00002600-\U000027BF"  # misc symbols, dingbats
    "\U0001F1E6-\U0001F1FF"  # regional indicator (flags)
    "\U00002700-\U000027BF"
    "\U0001F900-\U0001F9FF"
    "]+",
    flags=re.UNICODE,
)

# Small English profanity lexicon (~40 terms); near-zero on non-English rows, so confounded with `is_english`
PROFANITY_LEXICON = {
    "fuck", "fucking", "fucker", "fuckin", "shit", "shitty", "bitch", "bitches",
    "asshole", "ass", "bastard", "damn", "dumbass", "crap", "piss", "pissed",
    "dick", "dickhead", "cock", "pussy", "cunt", "slut", "whore", "faggot",
    "fag", "retard", "retarded", "moron", "idiot", "stupid", "scum", "trash",
    "loser", "freak", "psycho", "douche", "douchebag", "prick", "twat",
    "wanker", "bollocks",
}


def _tokenize_words(text):
    return text.lower().split()


def compute_text_features(texts):
    """Lexical / domain-specific descriptors computed from raw text alone."""
    n = len(texts)
    doc_word_lists = [_tokenize_words(t) for t in texts]
    doc_lens_words = np.array([len(w) for w in doc_word_lists], dtype=float)
    doc_lens_chars = np.array([len(t) for t in texts], dtype=float)

    vocab_counter = Counter()
    for w in doc_word_lists:
        vocab_counter.update(w)
    total_tokens = sum(vocab_counter.values())
    vocab_size = len(vocab_counter)
    hapax = sum(1 for c in vocab_counter.values() if c == 1)

    all_word_lens = [len(w) for wl in doc_word_lists for w in wl]
    mean_word_len = float(np.mean(all_word_lens)) if all_word_lens else 0.0

    n_mentions = np.array([len(MENTION_RE.findall(t)) for t in texts])
    n_hashtags = np.array([len(HASHTAG_RE.findall(t)) for t in texts])
    has_url = np.array([1 if URL_RE.search(t) else 0 for t in texts])
    has_emoji = np.array([1 if EMOJI_RE.search(t) else 0 for t in texts])
    n_excl = np.array([len(EXCLAMATION_RE.findall(t)) for t in texts])
    has_elong = np.array([1 if ELONGATION_RE.search(t) else 0 for t in texts])

    letters = "".join(ch for t in texts for ch in t if ch.isalpha())
    n_upper = sum(1 for ch in letters if ch.isupper())
    uppercase_char_ratio = (n_upper / len(letters)) if letters else 0.0

    profanity_hits_per_doc = []
    profanity_docs = 0
    total_profanity_tokens = 0
    for wl in doc_word_lists:
        stripped = [re.sub(r"[^\w]", "", w) for w in wl]
        hits = sum(1 for w in stripped if w in PROFANITY_LEXICON)
        profanity_hits_per_doc.append(hits)
        if hits > 0:
            profanity_docs += 1
        total_profanity_tokens += hits

    feats = {
        "vocab_size": vocab_size,
        "log_vocab_size": float(np.log10(max(vocab_size, 1))),
        "vocab_per_instance": vocab_size / n,
        "type_token_ratio": vocab_size / total_tokens if total_tokens else 0.0,
        "hapax_ratio": hapax / vocab_size if vocab_size else 0.0,
        "mean_doc_len_words": float(np.mean(doc_lens_words)),
        "std_doc_len_words": float(np.std(doc_lens_words)),
        "mean_doc_len_chars": float(np.mean(doc_lens_chars)),
        "mean_word_len": mean_word_len,
        "pct_docs_with_mention": float(np.mean(n_mentions > 0)),
        "mean_mentions_per_doc": float(np.mean(n_mentions)),
        "pct_docs_with_url": float(np.mean(has_url)),
        "pct_docs_with_hashtag": float(np.mean(n_hashtags > 0)),
        "mean_hashtags_per_doc": float(np.mean(n_hashtags)),
        "pct_docs_with_emoji": float(np.mean(has_emoji)),
        "uppercase_char_ratio": uppercase_char_ratio,
        "exclamation_rate": float(np.mean(n_excl)),
        "elongation_rate": float(np.mean(has_elong)),
        "pct_docs_with_profanity": profanity_docs / n,
        "profanity_token_rate": total_profanity_tokens / total_tokens if total_tokens else 0.0,
    }
    return feats


def compute_class_balance_features(labels):
    n = len(labels)
    counts = np.bincount(labels)
    counts = counts[counts > 0]
    n_classes = len(counts)
    majority_count = counts.max()
    minority_count = counts.min()
    probs = counts / n
    entropy_bits = float(-np.sum(probs * np.log2(probs)))
    return {
        "n_instances": n,
        "n_classes": n_classes,
        "log_instances": float(np.log10(n)),
        "instances_per_class": n / n_classes,
        "majority_pct": float(majority_count / n),
        "minority_pct": float(minority_count / n),
        "imbalance_ratio": float(majority_count / minority_count),
        "class_entropy": entropy_bits,
    }


def compute_mutual_info_features(texts, labels, random_state=42):
    """Mutual information between top-1000 TF-IDF terms and the label."""
    max_feats = 1000
    vec = TfidfVectorizer(max_features=max_feats, min_df=1)
    X = vec.fit_transform(texts)
    if X.shape[1] == 0:
        return {"mean_mutual_info": 0.0, "max_mutual_info": 0.0}
    # dense matrix needed by mutual_info_classif; capped at top-1000 terms
    mi = mutual_info_classif(X.toarray(), labels, discrete_features=False, random_state=random_state)
    return {
        "mean_mutual_info": float(np.mean(mi)),
        "max_mutual_info": float(np.max(mi)),
    }


def compute_embedding_features(embeddings, labels):
    """Separability / hardness descriptors in (cached, frozen) MiniLM space."""
    n = embeddings.shape[0]
    classes = np.unique(labels)

    # centroid_cosine_gap
    centroids = np.stack([embeddings[labels == c].mean(axis=0) for c in classes])
    if len(classes) >= 2:
        sim = cosine_similarity(centroids)
        iu = np.triu_indices(len(classes), k=1)
        mean_pair_sim = float(np.mean(sim[iu]))
        centroid_cosine_gap = 1.0 - mean_pair_sim
    else:
        centroid_cosine_gap = 0.0

    # knn1_disagreement: nearest neighbour (excluding self) label mismatch rate
    dist = cosine_distances(embeddings)
    np.fill_diagonal(dist, np.inf)
    nn_idx = np.argmin(dist, axis=1)
    disagreement = float(np.mean(labels[nn_idx] != labels))

    # silhouette_by_label
    if len(classes) >= 2 and n > len(classes):
        try:
            sil = float(silhouette_score(embeddings, labels, metric="cosine"))
        except ValueError:
            sil = 0.0
    else:
        sil = 0.0

    return {
        "centroid_cosine_gap": centroid_cosine_gap,
        "knn1_disagreement": disagreement,
        "silhouette_by_label": sil,
    }


def compute_manifest_features(language, platform):
    lang = (language or "").strip().lower()
    plat = (platform or "").strip().lower()
    return {
        "is_english": 1 if lang == "en" else 0,
        "is_multilingual_row": 1 if lang != "en" else 0,
        "is_twitter": 1 if plat == "twitter" else 0,
    }
