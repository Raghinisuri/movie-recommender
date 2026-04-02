"""
hybrid.py — Hybrid Recommender System.

Combines collaborative filtering (personalised to a user's rating history)
with content-based filtering (based on movie features) using a weighted blend.

Formula:
  hybrid_score = alpha * collab_score + (1 - alpha) * content_score

Both scores are normalised to [0, 1] before blending.
"""

import numpy as np
import pandas as pd
from collaborative import CollaborativeFilter
from content_based import ContentBasedFilter


class HybridRecommender:
    def __init__(self, alpha: float = 0.6, n_factors: int = 20):
        """
        alpha: weight given to collaborative filtering (0.0 = pure content, 1.0 = pure collab).
        """
        self.alpha = alpha
        self.collab = CollaborativeFilter(n_factors=n_factors)
        self.content = ContentBasedFilter()
        self.ratings_df = None
        self.movies_df = None

    def fit(self, ratings_df: pd.DataFrame, movies_df: pd.DataFrame):
        """Train both sub-models."""
        self.ratings_df = ratings_df
        self.movies_df = movies_df
        self.collab.fit(ratings_df)
        self.content.fit(movies_df)
        return self

    def recommend(self, user_id: int, n: int = 10) -> pd.DataFrame:
        """
        Generate hybrid recommendations for a user.
        Returns a DataFrame with columns: movieId, title, genres, hybrid_score,
        predicted_rating (collab), similarity_score (content).
        """
        # --- Collaborative recommendations (get more candidates, then blend) ---
        collab_recs = self.collab.recommend(user_id, self.movies_df, n=50)
        if collab_recs.empty:
            print("Falling back to content-based only (user not in training data).")
            liked = self._get_user_liked_movies(user_id, threshold=3.5)
            return self.content.recommend_for_user(liked, n=n)

        # --- Content-based recommendations based on user's highly rated movies ---
        liked_ids = self._get_user_liked_movies(user_id, threshold=3.5)
        already_rated = self._get_user_rated_movies(user_id)
        content_recs = self.content.recommend_for_user(liked_ids, n=50, exclude_ids=already_rated)

        # --- Normalise scores to [0, 1] ---
        def normalise(series):
            rng = series.max() - series.min()
            return (series - series.min()) / rng if rng > 0 else series * 0 + 0.5

        collab_scores = dict(zip(
            collab_recs["movieId"],
            normalise(collab_recs["predicted_rating"])
        ))
        content_scores = dict(zip(
            content_recs["movieId"],
            normalise(content_recs["similarity_score"])
        )) if not content_recs.empty else {}

        # --- Merge candidate pools ---
        all_ids = set(collab_scores) | set(content_scores)
        records = []
        for mid in all_ids:
            cs = collab_scores.get(mid, 0.0)
            ct = content_scores.get(mid, 0.0)
            hybrid = self.alpha * cs + (1 - self.alpha) * ct
            records.append({"movieId": mid, "collab_norm": cs, "content_norm": ct, "hybrid_score": hybrid})

        blended = pd.DataFrame(records).sort_values("hybrid_score", ascending=False).head(n)
        result = blended.merge(self.movies_df, on="movieId")

        # Re-attach original predicted rating
        if not collab_recs.empty:
            result = result.merge(
                collab_recs[["movieId", "predicted_rating"]],
                on="movieId", how="left"
            )

        cols = ["movieId", "title", "genres", "hybrid_score", "predicted_rating"]
        return result[[c for c in cols if c in result.columns]].reset_index(drop=True)

    def _get_user_liked_movies(self, user_id: int, threshold: float = 3.5) -> list:
        """Return movie IDs the user rated above threshold."""
        user_ratings = self.ratings_df[self.ratings_df["userId"] == user_id]
        liked = user_ratings[user_ratings["rating"] >= threshold]["movieId"].tolist()
        return liked if liked else user_ratings["movieId"].tolist()

    def _get_user_rated_movies(self, user_id: int) -> list:
        """Return all movie IDs the user has rated."""
        return self.ratings_df[self.ratings_df["userId"] == user_id]["movieId"].tolist()
