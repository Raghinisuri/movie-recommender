"""
main.py — Movie Recommendation System — Interactive CLI

Usage:
    python main.py

Commands at the prompt:
    recommend <user_id>          — Hybrid recommendations for a user
    collab <user_id>             — Collaborative filtering only
    similar <movie_id>           — Content-based: movies similar to this one
    search <title keyword>       — Search for a movie by title
    history <user_id>            — Show a user's rating history
    evaluate                     — Run full evaluation suite
    help                         — Show this menu
    quit                         — Exit
"""

import os
import sys
import pandas as pd
from sklearn.model_selection import train_test_split

# ── Ensure data exists ─────────────────────────────────────────────────────────
if not os.path.exists("data/ratings.csv"):
    print("Dataset not found. Generating synthetic data...")
    import generate_data  # noqa: F401

from collaborative import CollaborativeFilter
from content_based import ContentBasedFilter
from hybrid import HybridRecommender
from evaluation import (
    evaluate_collaborative,
    evaluate_ranking,
    catalog_coverage,
)

# ── Helpers ────────────────────────────────────────────────────────────────────
DIVIDER = "─" * 60


def banner():
    print("""
╔══════════════════════════════════════════════════════════╗
║        🎬  Movie Recommendation System                   ║
║        Collaborative + Content-Based + Hybrid            ║
╚══════════════════════════════════════════════════════════╝
""")


def fmt_movies(df: pd.DataFrame, score_col: str = None) -> str:
    if df.empty:
        return "  (no results)"
    lines = []
    for i, row in df.iterrows():
        score_str = ""
        if score_col and score_col in df.columns and pd.notna(row.get(score_col)):
            score_str = f"  [{score_col}: {row[score_col]:.2f}]"
        lines.append(f"  {i+1:2d}. {row['title']}")
        lines.append(f"       Genres: {row['genres']}{score_str}")
    return "\n".join(lines)


def load_data():
    ratings = pd.read_csv("data/ratings.csv")
    movies = pd.read_csv("data/movies.csv")
    return ratings, movies


def train_models(ratings, movies):
    train_df, test_df = train_test_split(ratings, test_size=0.2, random_state=42)

    print("Training collaborative filter (SVD)...")
    collab = CollaborativeFilter(n_factors=20)
    collab.fit(train_df)

    print("Building content-based filter (TF-IDF)...")
    content = ContentBasedFilter()
    content.fit(movies)

    print("Building hybrid recommender...")
    hybrid = HybridRecommender(alpha=0.6, n_factors=20)
    hybrid.fit(train_df, movies)

    print("Models ready.\n")
    return collab, content, hybrid, train_df, test_df


def run_evaluation(collab, hybrid, movies, test_df, ratings):
    print(f"\n{DIVIDER}")
    print("EVALUATION RESULTS")
    print(DIVIDER)

    print("\n▶  Collaborative Filter — Rating Prediction Accuracy")
    collab_metrics = evaluate_collaborative(collab, test_df)
    print(f"   RMSE           : {collab_metrics['RMSE']}")
    print(f"   MAE            : {collab_metrics['MAE']}")
    print(f"   Ratings tested : {collab_metrics['n_evaluated']}")

    print("\n▶  Hybrid Recommender — Ranking Quality (Precision & Recall @10)")
    ranking_metrics = evaluate_ranking(hybrid, test_df, movies, k=10, n_users=30)
    for k, v in ranking_metrics.items():
        print(f"   {k:<22}: {v}")

    print("\n▶  Catalog Coverage (% of movies recommended to ≥1 user)")
    sample_users = ratings["userId"].unique()[:50].tolist()
    cov = catalog_coverage(hybrid, movies["movieId"].tolist(), sample_users, n=10)
    print(f"   Coverage: {cov*100:.1f}%")
    print(DIVIDER)


def show_help():
    print("""
Commands:
  recommend <user_id>      Hybrid recommendations for a user
  collab    <user_id>      Collaborative filtering recommendations
  similar   <movie_id>     Movies similar to a given movie (content-based)
  search    <keyword>      Search movies by title
  history   <user_id>      Show user's rating history
  evaluate                 Run evaluation suite (RMSE, Precision@K, Coverage)
  users                    List available user IDs
  movies                   List all movies
  help                     Show this menu
  quit / exit              Exit
""")


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    banner()
    print("Loading data...")
    ratings, movies = load_data()
    print(f"Loaded {len(ratings)} ratings | {len(movies)} movies | {ratings['userId'].nunique()} users\n")

    collab, content, hybrid, train_df, test_df = train_models(ratings, movies)

    show_help()

    while True:
        try:
            raw = input(">> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not raw:
            continue

        parts = raw.split(maxsplit=1)
        cmd = parts[0].lower()
        arg = parts[1].strip() if len(parts) > 1 else ""

        # ── recommend ───────────────────────────────────────────────────────
        if cmd == "recommend":
            if not arg.isdigit():
                print("Usage: recommend <user_id>")
                continue
            uid = int(arg)
            print(f"\n🎬 Hybrid Recommendations for User {uid}\n{DIVIDER}")
            recs = hybrid.recommend(uid, n=10)
            print(fmt_movies(recs, score_col="hybrid_score"))
            print()

        # ── collab ──────────────────────────────────────────────────────────
        elif cmd == "collab":
            if not arg.isdigit():
                print("Usage: collab <user_id>")
                continue
            uid = int(arg)
            print(f"\n🤝 Collaborative Recommendations for User {uid}\n{DIVIDER}")
            recs = collab.recommend(uid, movies, n=10)
            print(fmt_movies(recs, score_col="predicted_rating"))
            print()

        # ── similar ─────────────────────────────────────────────────────────
        elif cmd == "similar":
            if not arg.isdigit():
                print("Usage: similar <movie_id>")
                continue
            mid = int(arg)
            movie_row = movies[movies["movieId"] == mid]
            if movie_row.empty:
                print(f"Movie ID {mid} not found.")
                continue
            title = movie_row.iloc[0]["title"]
            print(f"\n🎞  Movies similar to '{title}'\n{DIVIDER}")
            recs = content.get_similar_movies(mid, n=10)
            print(fmt_movies(recs, score_col="similarity"))
            print()

        # ── search ──────────────────────────────────────────────────────────
        elif cmd == "search":
            if not arg:
                print("Usage: search <keyword>")
                continue
            results = content.search_movie(arg)
            if results.empty:
                print(f"No movies found matching '{arg}'.")
            else:
                print(f"\n🔍 Search results for '{arg}'\n{DIVIDER}")
                for _, row in results.iterrows():
                    print(f"  ID {row['movieId']:4d} | {row['title']}  [{row['genres']}]")
            print()

        # ── history ─────────────────────────────────────────────────────────
        elif cmd == "history":
            if not arg.isdigit():
                print("Usage: history <user_id>")
                continue
            uid = int(arg)
            user_ratings = ratings[ratings["userId"] == uid].merge(movies, on="movieId")
            if user_ratings.empty:
                print(f"No ratings found for user {uid}.")
                continue
            user_ratings = user_ratings.sort_values("rating", ascending=False)
            print(f"\n📋 Rating History for User {uid} ({len(user_ratings)} movies)\n{DIVIDER}")
            for _, row in user_ratings.iterrows():
                stars = "★" * int(row["rating"]) + "☆" * (5 - int(row["rating"]))
                print(f"  {stars} ({row['rating']}) — {row['title']}")
            print()

        # ── evaluate ────────────────────────────────────────────────────────
        elif cmd == "evaluate":
            run_evaluation(collab, hybrid, movies, test_df, ratings)
            print()

        # ── users ───────────────────────────────────────────────────────────
        elif cmd == "users":
            uids = sorted(ratings["userId"].unique().tolist())
            print(f"\nAvailable User IDs: {uids[:50]}")
            if len(uids) > 50:
                print(f"  ... and {len(uids)-50} more")
            print()

        # ── movies ──────────────────────────────────────────────────────────
        elif cmd == "movies":
            print(f"\nAll Movies ({len(movies)} total)\n{DIVIDER}")
            for _, row in movies.iterrows():
                print(f"  ID {row['movieId']:4d} | {row['title']}  [{row['genres']}]")
            print()

        # ── help ────────────────────────────────────────────────────────────
        elif cmd == "help":
            show_help()

        # ── quit ────────────────────────────────────────────────────────────
        elif cmd in ("quit", "exit", "q"):
            print("Goodbye! 🎬")
            break

        else:
            print(f"Unknown command '{cmd}'. Type 'help' to see all commands.")


if __name__ == "__main__":
    main()
