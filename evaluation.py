"""
evaluation.py — Evaluation metrics for recommendation systems.

Metrics:
  - RMSE (Root Mean Square Error): how far predicted ratings are from actual.
  - MAE  (Mean Absolute Error): average absolute difference.
  - Precision@K: of the top-K recommendations, what fraction are relevant?
  - Recall@K: of all relevant movies, what fraction appear in top-K?
  - Coverage: % of the movie catalog the model can recommend.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def train_test_split_ratings(ratings_df: pd.DataFrame, test_size: float = 0.2, seed: int = 42):
    """Split ratings into train/test sets."""
    train, test = train_test_split(ratings_df, test_size=test_size, random_state=seed)
    return train.reset_index(drop=True), test.reset_index(drop=True)


def rmse(actual: list, predicted: list) -> float:
    a, p = np.array(actual), np.array(predicted)
    return float(np.sqrt(np.mean((a - p) ** 2)))


def mae(actual: list, predicted: list) -> float:
    a, p = np.array(actual), np.array(predicted)
    return float(np.mean(np.abs(a - p)))


def precision_at_k(recommended_ids: list, relevant_ids: set, k: int) -> float:
    """Fraction of top-K recommendations that are relevant."""
    top_k = recommended_ids[:k]
    hits = sum(1 for mid in top_k if mid in relevant_ids)
    return hits / k if k > 0 else 0.0


def recall_at_k(recommended_ids: list, relevant_ids: set, k: int) -> float:
    """Fraction of relevant movies found in top-K recommendations."""
    top_k = recommended_ids[:k]
    hits = sum(1 for mid in top_k if mid in relevant_ids)
    return hits / len(relevant_ids) if relevant_ids else 0.0


def evaluate_collaborative(model, test_df: pd.DataFrame) -> dict:
    """Evaluate collaborative filter on a test set."""
    preds, actuals = [], []
    for _, row in test_df.iterrows():
        pred = model.predict_rating(int(row["userId"]), int(row["movieId"]))
        if pred is not None:
            preds.append(pred)
            actuals.append(row["rating"])
    if not preds:
        return {"RMSE": None, "MAE": None, "n": 0}
    return {
        "RMSE": round(rmse(actuals, preds), 4),
        "MAE": round(mae(actuals, preds), 4),
        "n_evaluated": len(preds),
    }


def evaluate_ranking(recommender, test_df: pd.DataFrame, movies_df: pd.DataFrame,
                     k: int = 10, threshold: float = 3.5, n_users: int = 30) -> dict:
    """
    Evaluate ranking quality (Precision@K, Recall@K) for a sample of users.
    A movie is 'relevant' if the user rated it >= threshold in the test set.
    """
    p_at_k, r_at_k = [], []
    sample_users = test_df["userId"].unique()[:n_users]

    for uid in sample_users:
        user_test = test_df[test_df["userId"] == uid]
        relevant = set(user_test[user_test["rating"] >= threshold]["movieId"].tolist())
        if not relevant:
            continue

        try:
            recs = recommender.recommend(int(uid), n=k)
            if recs.empty:
                continue
            rec_ids = recs["movieId"].tolist()
            p_at_k.append(precision_at_k(rec_ids, relevant, k))
            r_at_k.append(recall_at_k(rec_ids, relevant, k))
        except Exception:
            continue

    return {
        f"Precision@{k}": round(np.mean(p_at_k), 4) if p_at_k else 0.0,
        f"Recall@{k}": round(np.mean(r_at_k), 4) if r_at_k else 0.0,
        "n_users_evaluated": len(p_at_k),
    }


def catalog_coverage(recommender, all_movie_ids: list,
                     sample_user_ids: list, n: int = 10) -> float:
    """Fraction of movies that appear in at least one user's recommendations."""
    recommended = set()
    for uid in sample_user_ids:
        try:
            recs = recommender.recommend(int(uid), n=n)
            recommended.update(recs["movieId"].tolist())
        except Exception:
            continue
    return round(len(recommended) / len(all_movie_ids), 4) if all_movie_ids else 0.0
