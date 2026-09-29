from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import ComplementNB
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier

RANDOM_STATE = 42

TEXT_ALGORITHMS = [
    "tfidf_word_lr",
    "tfidf_char_svm",
    "tfidf_nb",
    "svd_rf",
    "svd_histgb",
    "svd_knn",
]
EMBED_ALGORITHMS = [
    "minilm_lr",
    "minilm_mlp",
    "hatebert_lr",
    "mbert_lr",
]
ALL_ALGORITHMS = TEXT_ALGORITHMS + EMBED_ALGORITHMS

EMBED_MODEL_FOR_ALGO = {
    "minilm_lr": "minilm",
    "minilm_mlp": "minilm",
    "hatebert_lr": "hatebert",
    "mbert_lr": "mbert",
}


def _word_tfidf():
    return TfidfVectorizer(
        ngram_range=(1, 2), max_features=20000, min_df=2, sublinear_tf=True
    )


def make_text_pipeline(name, svd_n_components=200):
    """Build a fresh, unfitted sklearn Pipeline for one TF-IDF-family algorithm."""
    if name == "tfidf_word_lr":
        return Pipeline([
            ("tfidf", _word_tfidf()),
            ("clf", LogisticRegression(
                max_iter=2000, class_weight="balanced", random_state=RANDOM_STATE
            )),
        ])
    if name == "tfidf_char_svm":
        return Pipeline([
            ("tfidf", TfidfVectorizer(
                analyzer="char_wb", ngram_range=(3, 5), max_features=30000,
                min_df=2, sublinear_tf=True,
            )),
            ("clf", LinearSVC(class_weight="balanced", random_state=RANDOM_STATE)),
        ])
    if name == "tfidf_nb":
        return Pipeline([
            ("tfidf", TfidfVectorizer(ngram_range=(1, 2), max_features=20000, min_df=2)),
            ("clf", ComplementNB()),
        ])
    if name == "svd_rf":
        return Pipeline([
            ("tfidf", _word_tfidf()),
            ("svd", TruncatedSVD(n_components=svd_n_components, random_state=RANDOM_STATE)),
            ("clf", RandomForestClassifier(
                n_estimators=200, class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1
            )),
        ])
    if name == "svd_histgb":
        return Pipeline([
            ("tfidf", _word_tfidf()),
            ("svd", TruncatedSVD(n_components=svd_n_components, random_state=RANDOM_STATE)),
            ("clf", HistGradientBoostingClassifier(random_state=RANDOM_STATE)),
        ])
    if name == "svd_knn":
        return Pipeline([
            ("tfidf", _word_tfidf()),
            ("svd", TruncatedSVD(n_components=svd_n_components, random_state=RANDOM_STATE)),
            ("clf", KNeighborsClassifier(n_neighbors=5, metric="cosine", n_jobs=-1)),
        ])
    raise ValueError(f"unknown text algorithm {name!r}")


def make_embedding_classifier(name):
    """Classifier head for one frozen-embedding algorithm (scaler fit inside the fold)."""
    if name in ("minilm_lr", "hatebert_lr", "mbert_lr"):
        clf = LogisticRegression(max_iter=2000, class_weight="balanced", random_state=RANDOM_STATE)
    elif name == "minilm_mlp":
        clf = MLPClassifier(hidden_layer_sizes=(128,), max_iter=500, random_state=RANDOM_STATE)
    else:
        raise ValueError(f"unknown embedding algorithm {name!r}")
    return Pipeline([("scaler", StandardScaler()), ("clf", clf)])
