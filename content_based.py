"""
content_based.py — Content-Based Filtering using TF-IDF + Cosine Similarity.

How it works:
  1. Represent each movie as a "document" of its genres.
  2. Compute TF-IDF vectors for all movies.
  3. Calculate pairwise cosine similarity.
  4. For a given movie (or a user's watch history), find the most similar movies.
"""

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class ContentBasedFilter:
    def __init__(self):
        self.tfidf = TfidfVectorizer(token_pattern=r"[^|]+")
        self.movies_df = None
        self.cosine_sim = None
        self.index_map = {}   # movieId -> matrix row index
        self.reverse_map = {} # row index -> movieId

    def fit(self, movies_df: pd.DataFrame):
        """
        Build TF-IDF matrix and pairwise cosine similarity from movie genres.
        movies_df columns: movieId, title, genres
        """
        self.movies_df = movies_df.reset_index(drop=True).copy()

        # Build index maps
        for idx, row in self.movies_df.iterrows():
            self.index_map[row["movieId"]] = idx
            self.reverse_map[idx] = row["movieId"]

        # TF-IDF on genres (pipe-separated: "Action|Thriller|Drama")
        tfidf_matrix = self.tfidf.fit_transform(self.movies_df["genres"])

        # Pairwise cosine similarity: shape = (n_movies, n_movies)
        self.cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
        return self

    def get_similar_movies(self, movie_id: int, n: int = 10) -> pd.DataFrame:
        """Find the n most similar movies to a given movie."""
        if movie_id not in self.index_map:
            print(f"Movie ID {movie_id} not found.")
            return pd.DataFrame()

        idx = self.index_map[movie_id]
        sim_scores = list(enumerate(self.cosine_sim[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        # Skip the movie itself (score = 1.0 with itself)
        sim_scores = [(i, s) for i, s in sim_scores if i != idx][:n]

        movie_indices = [i for i, _ in sim_scores]
        scores = {i: round(s, 4) for i, s in sim_scores}

        result = self.movies_df.iloc[movie_indices].copy()
        result["similarity"] = result.index.map(scores)
        return result.reset_index(drop=True)

    def recommend_for_user(
        self,
        liked_movie_ids: list,
        n: int = 10,
        exclude_ids: list = None
    ) -> pd.DataFrame:
        """
        Recommend movies based on a user's list of liked movie IDs.
        Aggregates similarity scores across all liked movies.
        exclude_ids: movie IDs to exclude (e.g., already watched).
        """
        exclude = set(exclude_ids or []) | set(liked_movie_ids)
        score_agg = {}

        for mid in liked_movie_ids:
            if mid not in self.index_map:
                continue
            idx = self.index_map[mid]
            for j, sim in enumerate(self.cosine_sim[idx]):
                candidate_id = self.reverse_map[j]
                if candidate_id in exclude:
                    continue
                score_agg[candidate_id] = score_agg.get(candidate_id, 0) + sim

        if not score_agg:
            return pd.DataFrame()

        top_ids = sorted(score_agg, key=score_agg.get, reverse=True)[:n]
        top_scores = {mid: round(score_agg[mid], 4) for mid in top_ids}

        result = self.movies_df[self.movies_df["movieId"].isin(top_ids)].copy()
        result["similarity_score"] = result["movieId"].map(top_scores)
        return result.sort_values("similarity_score", ascending=False).reset_index(drop=True)

    def get_movie_genres(self, movie_id: int) -> str:
        """Return genres string for a given movie ID."""
        row = self.movies_df[self.movies_df["movieId"] == movie_id]
        if row.empty:
            return "Unknown"
        return row.iloc[0]["genres"]

    def search_movie(self, query: str) -> pd.DataFrame:
        """Fuzzy title search."""
        query = query.lower()
        mask = self.movies_df["title"].str.lower().str.contains(query, na=False)
        return self.movies_df[mask][["movieId", "title", "genres"]].reset_index(drop=True)
