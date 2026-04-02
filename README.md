# 🎬 Movie Recommendation System

A complete Python implementation of a Movie Recommendation System using:
- **Collaborative Filtering** — Matrix Factorization via SVD (Singular Value Decomposition)
- **Content-Based Filtering** — TF-IDF vectors + Cosine Similarity on movie genres
- **Hybrid Recommender** — Weighted blend of both approaches
- **Evaluation** — RMSE, MAE, Precision@K, Recall@K, Catalog Coverage

---

## Project Structure

```
movie_recommender/
├── main.py            # Interactive CLI — run this to start
├── collaborative.py   # Collaborative filtering (SVD matrix factorization)
├── content_based.py   # Content-based filtering (TF-IDF + cosine similarity)
├── hybrid.py          # Hybrid recommender (blends both approaches)
├── evaluation.py      # Evaluation metrics (RMSE, Precision@K, Coverage)
├── generate_data.py   # Generates synthetic MovieLens-style dataset
├── data/
│   ├── ratings.csv    # userId, movieId, rating
│   └── movies.csv     # movieId, title, genres
└── README.md
```

---

## Quick Start

```bash
# 1. Install dependencies
pip install pandas numpy scikit-learn

# 2. Generate dataset (auto-runs on first start, or run manually)
python generate_data.py

# 3. Launch the interactive recommender
python main.py
```

---

## CLI Commands

| Command | Description |
|---|---|
| `recommend <user_id>` | Hybrid top-10 recommendations for a user |
| `collab <user_id>` | Collaborative filtering recommendations only |
| `similar <movie_id>` | Find movies similar to a given movie (content-based) |
| `search <keyword>` | Search movies by title |
| `history <user_id>` | View a user's rating history |
| `evaluate` | Run full evaluation suite |
| `users` | List all available user IDs |
| `movies` | List all movies with IDs |
| `help` | Show all commands |
| `quit` | Exit |

**Example session:**
```
>> search dark knight
  ID   40 | The Dark Knight (2008)  [Action|Crime|Drama|Thriller]

>> similar 40
  1. Inception (2010)          [Action|Adventure|Sci-Fi|Thriller]  similarity: 0.71
  2. The Matrix (1999)         [Action|Sci-Fi|Thriller]            similarity: 0.68
  ...

>> recommend 1
  1. Forrest Gump (1994)       [Comedy|Drama|Romance|War]  hybrid_score: 0.91
  2. The Shawshank Redemption  [Crime|Drama]               hybrid_score: 0.88
  ...

>> evaluate
  RMSE: 0.7832
  Precision@10: 0.2145
  Coverage: 72.0%
```

---

## How Each Algorithm Works

### Collaborative Filtering (SVD)

Builds a User × Movie matrix of ratings, then decomposes it into latent factors using Truncated SVD:

```
R ≈ U × Σ × Vt
```

- **U**: user-factor matrix (what each user "is like")
- **Σ**: strength of each latent factor
- **Vt**: movie-factor matrix (what each movie "is like")

Missing ratings are predicted by reconstructing the matrix from these factors. The model learns patterns like "users who love action thrillers also tend to like sci-fi" without being explicitly told that.

### Content-Based Filtering (TF-IDF + Cosine Similarity)

Represents each movie as a TF-IDF vector of its genres, then computes pairwise cosine similarity:

```
similarity(A, B) = (A · B) / (|A| × |B|)
```

Movies with overlapping genres score high similarity. Recommendations are driven by what a user has already liked.

### Hybrid Recommender

Blends both scores with a tunable alpha parameter:

```
hybrid_score = alpha × collab_score + (1 - alpha) × content_score
```

Default: `alpha = 0.6` (60% collaborative, 40% content-based).

### Evaluation Metrics

| Metric | What it measures |
|---|---|
| RMSE | Accuracy of predicted ratings vs actual ratings |
| MAE | Same but less sensitive to outliers |
| Precision@K | Of top-K recommendations, what fraction are relevant? |
| Recall@K | Of all relevant movies, what fraction appear in top-K? |
| Coverage | What % of the catalog gets recommended to any user? |

---

## Using Your Own Data (Real MovieLens)

1. Download from [https://grouplens.org/datasets/movielens/](https://grouplens.org/datasets/movielens/)
2. Replace `data/ratings.csv` and `data/movies.csv` with the real files
3. Make sure the columns match: `userId, movieId, rating` and `movieId, title, genres`

The MovieLens 100K dataset has 100,000 ratings from 943 users across 1,682 movies.

---

## Dependencies

- `pandas` — data loading and manipulation
- `numpy` — matrix operations
- `scikit-learn` — TruncatedSVD, TF-IDF, cosine similarity, train/test split

No external recommendation libraries needed — the SVD is implemented using scikit-learn's decomposition module.
