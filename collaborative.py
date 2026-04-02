"""
collaborative.py — Collaborative Filtering via Matrix Factorization (SVD).

How it works:
  1. Build a User x Movie ratings matrix (sparse — most entries are 0/unknown).
  2. Decompose it into latent factors using Truncated SVD.
  3. Reconstruct the full matrix to predict missing ratings.
  4. Recommend movies with the highest predicted ratings the user hasn't seen.
"""

import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import mean_squared_error


class CollaborativeFilter:
    def __init__(self, n_factors: int = 20):
        """
        n_factors: number of latent factors (higher = more expressive, but slower).
        """
        self.n_factors = n_factors
        self.svd = TruncatedSVD(n_components=n_factors, random_state=42)
        self.user_item_matrix = None
        self.user_ids = None
        self.movie_ids = None
        self.predicted_matrix = None
        self.user_index = {}
        self.movie_index = {}

    def fit(self, ratings_df: pd.DataFrame):
        """
        Build and decompose the user-item matrix.
        ratings_df columns: userId, movieId, rating
        """
        # Pivot ratings into a User x Movie matrix
        matrix = ratings_df.pivot_table(
            index="userId", columns="movieId", values="rating", fill_value=0
        )
        self.user_item_matrix = matrix
        self.user_ids = matrix.index.tolist()
        self.movie_ids = matrix.columns.tolist()
        self.user_index = {uid: i for i, uid in enumerate(self.user_ids)}
        self.movie_index = {mid: j for j, mid in enumerate(self.movie_ids)}

        # Normalise: subtract each user's mean rating (so 0 means "not rated", not "rated 0")
        mat = matrix.values.astype(float)
        self.user_means = np.true_divide(
            mat.sum(1), (mat != 0).sum(1).clip(min=1)
        )
        mat_centered = mat.copy()
        for i in range(mat.shape[0]):
            mask = mat[i] != 0
            mat_centered[i][mask] -= self.user_means[i]

        # SVD decomposition: mat ≈ U * Σ * Vt
        self.U = self.svd.fit_transform(mat_centered)
        self.Vt = self.svd.components_
        self.sigma = np.diag(self.svd.singular_values_)

        # Reconstruct full matrix and add back user means
        reconstructed = self.U @ self.sigma @ self.Vt
        for i in range(reconstructed.shape[0]):
            reconstructed[i] += self.user_means[i]

        self.predicted_matrix = np.clip(reconstructed, 0.5, 5.0)
        return self

    def predict_rating(self, user_id: int, movie_id: int) -> float:
        """Predict the rating user_id would give movie_id."""
        if user_id not in self.user_index or movie_id not in self.movie_index:
            return None
        i = self.user_index[user_id]
        j = self.movie_index[movie_id]
        return round(self.predicted_matrix[i, j] * 2) / 2

    def recommend(self, user_id: int, movies_df: pd.DataFrame, n: int = 10) -> pd.DataFrame:
        """
        Return top-n movie recommendations for user_id,
        excluding movies they've already rated.
        """
        if user_id not in self.user_index:
            print(f"User {user_id} not found in training data.")
            return pd.DataFrame()

        i = self.user_index[user_id]
        already_rated = set(
            j for j, mid in enumerate(self.movie_ids)
            if self.user_item_matrix.iloc[i, j] != 0
        )

        scores = []
        for j, mid in enumerate(self.movie_ids):
            if j not in already_rated:
                scores.append((mid, self.predicted_matrix[i, j]))

        scores.sort(key=lambda x: x[1], reverse=True)
        top_ids = [s[0] for s in scores[:n]]
        top_scores = {s[0]: round(s[1], 2) for s in scores[:n]}

        result = movies_df[movies_df["movieId"].isin(top_ids)].copy()
        result["predicted_rating"] = result["movieId"].map(top_scores)
        return result.sort_values("predicted_rating", ascending=False).reset_index(drop=True)

    def evaluate(self, test_df: pd.DataFrame) -> dict:
        """Compute RMSE and MAE on a held-out test set."""
        preds, actuals = [], []
        for _, row in test_df.iterrows():
            pred = self.predict_rating(int(row["userId"]), int(row["movieId"]))
            if pred is not None:
                preds.append(pred)
                actuals.append(row["rating"])

        if not preds:
            return {"RMSE": None, "MAE": None, "n_evaluated": 0}

        rmse = np.sqrt(mean_squared_error(actuals, preds))
        mae = np.mean(np.abs(np.array(actuals) - np.array(preds)))
        return {"RMSE": round(rmse, 4), "MAE": round(mae, 4), "n_evaluated": len(preds)}
