"""
app.py — Streamlit Web App for Movie Recommendation System
Run locally with: streamlit run app.py
"""

import streamlit as st
import pandas as pd
import os
from sklearn.model_selection import train_test_split

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="🎬 Movie Recommender",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-title {
        font-size: 2.5rem;
        font-weight: 700;
        color: #E50914;
        text-align: center;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1rem;
        color: #888;
        text-align: center;
        margin-bottom: 2rem;
    }
    .movie-card {
        background: #1a1a2e;
        border-radius: 12px;
        padding: 1rem 1.2rem;
        margin-bottom: 0.6rem;
        border-left: 4px solid #E50914;
    }
    .movie-title {
        font-size: 1rem;
        font-weight: 600;
        color: #ffffff;
        margin: 0;
    }
    .movie-genre {
        font-size: 0.8rem;
        color: #aaaaaa;
        margin: 0.2rem 0 0 0;
    }
    .score-badge {
        background: #E50914;
        color: white;
        border-radius: 20px;
        padding: 2px 10px;
        font-size: 0.8rem;
        font-weight: 600;
    }
    .metric-card {
        background: #1a1a2e;
        border-radius: 10px;
        padding: 1rem;
        text-align: center;
    }
    .stButton>button {
        background-color: #E50914;
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.5rem 2rem;
        font-size: 1rem;
        font-weight: 600;
        width: 100%;
    }
    .stButton>button:hover {
        background-color: #b20710;
    }
</style>
""", unsafe_allow_html=True)


# ── Load models (cached so they only train once) ───────────────────────────────
@st.cache_resource
def load_models():
    if not os.path.exists("data/ratings.csv"):
        import generate_data  # noqa

    from collaborative import CollaborativeFilter
    from content_based import ContentBasedFilter
    from hybrid import HybridRecommender

    ratings = pd.read_csv("data/ratings.csv")
    movies  = pd.read_csv("data/movies.csv")
    train, test = train_test_split(ratings, test_size=0.2, random_state=42)

    collab  = CollaborativeFilter(n_factors=20).fit(train)
    content = ContentBasedFilter().fit(movies)
    hybrid  = HybridRecommender(alpha=0.6).fit(train, movies)

    return ratings, movies, collab, content, hybrid, test


# ── Helper: render movie cards ─────────────────────────────────────────────────
def render_movie_cards(df, score_col=None, score_label="Score"):
    if df.empty:
        st.warning("No results found.")
        return
    for i, row in df.iterrows():
        score_str = ""
        if score_col and score_col in df.columns and pd.notna(row.get(score_col)):
            score_str = f'<span class="score-badge">{score_label}: {row[score_col]:.2f}</span>'
        st.markdown(f"""
        <div class="movie-card">
            <p class="movie-title">#{i+1} &nbsp; {row['title']} &nbsp; {score_str}</p>
            <p class="movie-genre">🎭 {row['genres']}</p>
        </div>
        """, unsafe_allow_html=True)


# ── Main app ───────────────────────────────────────────────────────────────────
def main():
    # Header
    st.markdown('<p class="main-title">🎬 Movie Recommendation System</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-title">Collaborative Filtering + Content-Based + Hybrid AI</p>', unsafe_allow_html=True)

    # Load
    with st.spinner("Loading models... (first load takes ~10 seconds)"):
        ratings, movies, collab, content, hybrid, test = load_models()

    # Sidebar
    st.sidebar.image("https://img.icons8.com/color/96/cinema-.png", width=80)
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", [
        "🎯 Get Recommendations",
        "🎞 Similar Movies",
        "📋 User History",
        "📊 Evaluation",
        "🎬 All Movies",
    ])

    st.sidebar.markdown("---")
    st.sidebar.markdown(f"**Dataset:**\n- {len(ratings)} ratings\n- {len(movies)} movies\n- {ratings['userId'].nunique()} users")

    # ── Page: Recommendations ────────────────────────────────────────────────
    if page == "🎯 Get Recommendations":
        st.header("🎯 Get Movie Recommendations")
        st.write("Select a user and get personalised movie recommendations powered by Hybrid AI.")

        col1, col2 = st.columns([2, 1])
        with col1:
            user_id = st.selectbox(
                "Select User ID",
                sorted(ratings["userId"].unique().tolist()),
                index=0
            )
        with col2:
            n_recs = st.slider("Number of recommendations", 5, 20, 10)

        mode = st.radio("Recommendation mode", ["Hybrid (Best)", "Collaborative Only"], horizontal=True)

        if st.button("🎬 Get Recommendations"):
            with st.spinner("Finding best movies for you..."):
                if mode == "Hybrid (Best)":
                    recs = hybrid.recommend(user_id, n=n_recs)
                    score_col, score_label = "hybrid_score", "Score"
                else:
                    recs = collab.recommend(user_id, movies, n=n_recs)
                    score_col, score_label = "predicted_rating", "Predicted ★"

            st.success(f"Top {n_recs} recommendations for User {user_id}")
            render_movie_cards(recs, score_col=score_col, score_label=score_label)

    # ── Page: Similar Movies ─────────────────────────────────────────────────
    elif page == "🎞 Similar Movies":
        st.header("🎞 Find Similar Movies")
        st.write("Pick a movie and find others with similar genres using Content-Based Filtering.")

        movie_options = {f"{row['title']} (ID: {row['movieId']})": row["movieId"]
                        for _, row in movies.iterrows()}

        selected = st.selectbox("Select a Movie", list(movie_options.keys()))
        n_sim = st.slider("Number of similar movies", 5, 20, 10)

        if st.button("🔍 Find Similar Movies"):
            mid = movie_options[selected]
            with st.spinner("Finding similar movies..."):
                sims = content.get_similar_movies(mid, n=n_sim)

            st.success(f"Movies similar to **{selected.split(' (ID')[0]}**")
            render_movie_cards(sims, score_col="similarity", score_label="Similarity")

    # ── Page: User History ───────────────────────────────────────────────────
    elif page == "📋 User History":
        st.header("📋 User Rating History")
        st.write("See what movies a user has already rated.")

        user_id = st.selectbox("Select User ID", sorted(ratings["userId"].unique().tolist()))

        if st.button("📋 Show History"):
            user_ratings = (
                ratings[ratings["userId"] == user_id]
                .merge(movies, on="movieId")
                .sort_values("rating", ascending=False)
            )
            st.success(f"User {user_id} has rated {len(user_ratings)} movies")

            for _, row in user_ratings.iterrows():
                stars = "⭐" * int(row["rating"])
                st.markdown(f"""
                <div class="movie-card">
                    <p class="movie-title">{stars} ({row['rating']}) &nbsp; {row['title']}</p>
                    <p class="movie-genre">🎭 {row['genres']}</p>
                </div>
                """, unsafe_allow_html=True)

    # ── Page: Evaluation ─────────────────────────────────────────────────────
    elif page == "📊 Evaluation":
        st.header("📊 Model Evaluation")
        st.write("Performance metrics for the recommendation system.")

        if st.button("▶ Run Evaluation"):
            from evaluation import evaluate_collaborative, evaluate_ranking, catalog_coverage

            with st.spinner("Running evaluation..."):
                collab_metrics = evaluate_collaborative(collab, test)
                ranking_metrics = evaluate_ranking(hybrid, test, movies, k=10, n_users=30)
                sample_users = ratings["userId"].unique()[:50].tolist()
                cov = catalog_coverage(hybrid, movies["movieId"].tolist(), sample_users)

            st.subheader("Collaborative Filter — Rating Accuracy")
            c1, c2, c3 = st.columns(3)
            c1.metric("RMSE", collab_metrics["RMSE"], help="Lower is better")
            c2.metric("MAE", collab_metrics["MAE"], help="Lower is better")
            c3.metric("Ratings Tested", collab_metrics["n_evaluated"])

            st.subheader("Hybrid Recommender — Ranking Quality")
            c4, c5, c6 = st.columns(3)
            c4.metric("Precision@10", f"{ranking_metrics['Precision@10']:.2%}", help="Higher is better")
            c5.metric("Recall@10", f"{ranking_metrics['Recall@10']:.2%}", help="Higher is better")
            c6.metric("Catalog Coverage", f"{cov*100:.1f}%", help="Higher is better")

            st.info("""
            **What these mean:**
            - **RMSE / MAE**: How accurately the model predicts ratings (lower = better)
            - **Precision@10**: Of top 10 recommendations, how many are actually relevant
            - **Recall@10**: Of all relevant movies, how many appear in top 10
            - **Coverage**: What % of all movies get recommended to at least one user
            """)

    # ── Page: All Movies ─────────────────────────────────────────────────────
    elif page == "🎬 All Movies":
        st.header("🎬 All Movies in Dataset")

        search = st.text_input("🔍 Search by title", "")
        if search:
            filtered = movies[movies["title"].str.lower().str.contains(search.lower(), na=False)]
        else:
            filtered = movies

        st.write(f"Showing {len(filtered)} movies")
        for _, row in filtered.iterrows():
            st.markdown(f"""
            <div class="movie-card">
                <p class="movie-title">ID {row['movieId']} &nbsp; {row['title']}</p>
                <p class="movie-genre">🎭 {row['genres']}</p>
            </div>
            """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
