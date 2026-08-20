from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler

from src.features.achievement_lexicon import AOSP_COLUMN_NAMES, extract_aosp_dataframe


def build_text_feature_matrix(
    train_texts: pd.Series,
    test_texts: pd.Series | None = None,
    max_features: int = 2000,
) -> tuple[np.ndarray, np.ndarray | None, TfidfVectorizer, np.ndarray, np.ndarray | None]:
    vectorizer = TfidfVectorizer(max_features=max_features, ngram_range=(1, 2), stop_words="english")
    tfidf_train = vectorizer.fit_transform(train_texts.fillna(""))
    tfidf_train_arr = np.empty((tfidf_train.shape[0], tfidf_train.shape[1]), dtype=np.float32)
    tfidf_train_arr[:] = tfidf_train.toarray()
    del tfidf_train
    aosp_train = extract_aosp_dataframe(train_texts).values.astype(np.float32)
    X_train = np.hstack([tfidf_train_arr, aosp_train])
    del tfidf_train_arr

    X_test = None
    if test_texts is not None:
        tfidf_test = vectorizer.transform(test_texts.fillna(""))
        tfidf_test_arr = np.empty((tfidf_test.shape[0], tfidf_test.shape[1]), dtype=np.float32)
        tfidf_test_arr[:] = tfidf_test.toarray()
        del tfidf_test
        aosp_test = extract_aosp_dataframe(test_texts).values.astype(np.float32)
        X_test = np.hstack([tfidf_test_arr, aosp_test])
        del tfidf_test_arr

    return X_train, X_test, vectorizer, aosp_train, None


def build_aosp_only_matrix(texts: pd.Series) -> np.ndarray:
    return extract_aosp_dataframe(texts)[AOSP_COLUMN_NAMES].values


def scale_features(X_train: np.ndarray, X_test: np.ndarray | None = None):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test) if X_test is not None else None
    return X_train_scaled, X_test_scaled, scaler


def compute_paradox_statistics(df: pd.DataFrame) -> dict[str, float]:
    """
    Quantify the idealised self-presentation paradox:
    high AOSP may coexist with well-being decline.
    """
    aosp = df["aosp_composite"].values
    distress = df.get("wellbeing_decline", df.get("distress_score", pd.Series(0))).values
    if "GAD_7_Score" in df.columns:
        distress = df["GAD_7_Score"].values + df["PHQ_9_Score"].values

    paradox = df["paradox_index"].values if "paradox_index" in df.columns else aosp * distress

    high_aosp = aosp >= np.median(aosp)
    high_distress = distress >= np.median(distress) if distress.dtype != int else distress == 1

    paradox_group_rate = (high_aosp & high_distress).mean()
    correlation = stats.pearsonr(aosp, distress)[0] if len(aosp) > 2 else 0.0
    paradox_corr = stats.pearsonr(aosp, paradox)[0] if len(paradox) > 2 else 0.0

    return {
        "aosp_distress_correlation": float(correlation),
        "paradox_index_correlation": float(paradox_corr),
        "high_aosp_high_distress_rate": float(paradox_group_rate),
        "mean_aosp": float(np.mean(aosp)),
        "mean_distress": float(np.mean(distress)),
        "mean_paradox_index": float(np.mean(paradox)),
    }
