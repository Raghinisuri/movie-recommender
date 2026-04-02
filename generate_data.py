"""
generate_data.py — Creates a realistic synthetic MovieLens-style dataset.
Run this once before using the recommender.
"""
import numpy as np
import pandas as pd
import os

np.random.seed(42)

MOVIES = [
    (1, "Toy Story (1995)", "Animation|Children|Comedy"),
    (2, "Jumanji (1995)", "Adventure|Children|Fantasy"),
    (3, "Grumpier Old Men (1995)", "Comedy|Romance"),
    (4, "Waiting to Exhale (1995)", "Comedy|Drama|Romance"),
    (5, "Father of the Bride Part II (1995)", "Comedy"),
    (6, "Heat (1995)", "Action|Crime|Thriller"),
    (7, "Sabrina (1995)", "Comedy|Romance"),
    (8, "Tom and Huck (1995)", "Adventure|Children"),
    (9, "Sudden Death (1995)", "Action"),
    (10, "GoldenEye (1995)", "Action|Adventure|Thriller"),
    (11, "American President, The (1995)", "Comedy|Drama|Romance"),
    (12, "Dracula: Dead and Loving It (1995)", "Comedy|Horror"),
    (13, "Balto (1995)", "Adventure|Animation|Children"),
    (14, "Nixon (1995)", "Drama"),
    (15, "Cutthroat Island (1995)", "Action|Adventure|Romance"),
    (16, "Casino (1995)", "Crime|Drama"),
    (17, "Sense and Sensibility (1995)", "Drama|Romance"),
    (18, "Four Rooms (1995)", "Comedy|Mystery|Thriller"),
    (19, "Ace Ventura: When Nature Calls (1995)", "Comedy"),
    (20, "Money Train (1995)", "Action|Comedy|Crime|Drama|Thriller"),
    (21, "Get Shorty (1995)", "Comedy|Crime|Thriller"),
    (22, "Copycat (1995)", "Crime|Drama|Horror|Mystery|Thriller"),
    (23, "Assassins (1995)", "Action|Crime|Thriller"),
    (24, "Powder (1995)", "Drama|Sci-Fi"),
    (25, "Leaving Las Vegas (1995)", "Drama|Romance"),
    (26, "Othello (1995)", "Drama"),
    (27, "Now and Then (1995)", "Children|Drama"),
    (28, "Persuasion (1995)", "Drama|Romance"),
    (29, "City of Lost Children, The (1995)", "Adventure|Drama|Fantasy|Mystery|Sci-Fi"),
    (30, "Shanghai Triad (1995)", "Crime|Drama"),
    (31, "Dangerous Minds (1995)", "Drama"),
    (32, "12 Angry Men (1957)", "Drama"),
    (33, "Forrest Gump (1994)", "Comedy|Drama|Romance|War"),
    (34, "Shawshank Redemption, The (1994)", "Crime|Drama"),
    (35, "Pulp Fiction (1994)", "Comedy|Crime|Drama|Thriller"),
    (36, "Schindler's List (1993)", "Drama|War"),
    (37, "Star Wars: Episode IV (1977)", "Action|Adventure|Sci-Fi"),
    (38, "Silence of the Lambs, The (1991)", "Crime|Horror|Thriller"),
    (39, "Fargo (1996)", "Comedy|Crime|Drama|Thriller"),
    (40, "The Dark Knight (2008)", "Action|Crime|Drama|Thriller"),
    (41, "Inception (2010)", "Action|Adventure|Sci-Fi|Thriller"),
    (42, "The Matrix (1999)", "Action|Sci-Fi|Thriller"),
    (43, "Goodfellas (1990)", "Crime|Drama"),
    (44, "Fight Club (1999)", "Drama|Thriller"),
    (45, "The Godfather (1972)", "Crime|Drama"),
    (46, "Interstellar (2014)", "Adventure|Drama|Sci-Fi"),
    (47, "The Lord of the Rings: The Fellowship (2001)", "Adventure|Fantasy"),
    (48, "Avengers: Endgame (2019)", "Action|Adventure|Sci-Fi"),
    (49, "Spirited Away (2001)", "Animation|Adventure|Fantasy"),
    (50, "Your Name (2016)", "Animation|Drama|Fantasy|Romance"),
]

NUM_USERS = 200
NUM_RATINGS_PER_USER = (15, 40)

genre_prefs = {
    "action_fan":      {"Action": 1.5, "Adventure": 1.2, "Thriller": 1.2, "Sci-Fi": 1.0},
    "drama_fan":       {"Drama": 1.5, "Romance": 1.2, "War": 1.0, "Crime": 1.0},
    "comedy_fan":      {"Comedy": 1.5, "Animation": 1.0, "Children": 1.0, "Romance": 1.0},
    "horror_fan":      {"Horror": 1.5, "Thriller": 1.2, "Crime": 1.0, "Mystery": 1.0},
    "scifi_fan":       {"Sci-Fi": 1.5, "Action": 1.2, "Adventure": 1.2, "Fantasy": 1.0},
    "animation_fan":   {"Animation": 1.5, "Children": 1.3, "Fantasy": 1.2, "Adventure": 1.0},
}

user_types = list(genre_prefs.keys())

def movie_genre_score(genres_str, prefs):
    genres = genres_str.split("|")
    score = sum(prefs.get(g, 0.7) for g in genres) / len(genres)
    return score

rows = []
for user_id in range(1, NUM_USERS + 1):
    utype = user_types[(user_id - 1) % len(user_types)]
    prefs = genre_prefs[utype]
    base_bias = np.random.uniform(-0.5, 0.5)
    n_ratings = np.random.randint(*NUM_RATINGS_PER_USER)
    movie_indices = np.random.choice(len(MOVIES), size=min(n_ratings, len(MOVIES)), replace=False)
    for idx in movie_indices:
        mid, title, genres = MOVIES[idx]
        score = movie_genre_score(genres, prefs)
        raw = score * 4.0 + base_bias + np.random.normal(0, 0.5)
        rating = round(max(0.5, min(5.0, raw)) * 2) / 2
        rows.append({"userId": user_id, "movieId": mid, "rating": rating})

ratings_df = pd.DataFrame(rows)
movies_df = pd.DataFrame(MOVIES, columns=["movieId", "title", "genres"])

os.makedirs("data", exist_ok=True)
ratings_df.to_csv("data/ratings.csv", index=False)
movies_df.to_csv("data/movies.csv", index=False)

print(f"Generated {len(ratings_df)} ratings from {NUM_USERS} users across {len(MOVIES)} movies.")
print(f"Saved to data/ratings.csv and data/movies.csv")
