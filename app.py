import os
import time

import numpy as np
import pandas as pd
from flask import Flask, jsonify, render_template, request
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split

app = Flask(__name__)
# Bisa diubah sesuai data yang dibutuhkan
RATING_SAMPLE_SIZE = 20_000_263

# Threshold rating: film dianggap "disukai" jika rating >= LIKE_THRESHOLD
LIKE_THRESHOLD = 4.0
TEST_SIZE = 0.2
RANDOM_STATE = 42

def load_dataset(movie_path: str, rating_path: str, sample_size: int = None):
    print("=" * 60)
    print("STEP 1: Load Dataset")
    print("=" * 60)

    movies_df = pd.read_csv(movie_path)
    print(f"[INFO] movie.csv dimuat: {movies_df.shape[0]} baris, {movies_df.shape[1]} kolom")
    print(f"       Kolom: {list(movies_df.columns)}")

    if sample_size is not None:
        print(f"[INFO] Memuat rating.csv dengan sampling {sample_size:,} baris...")
        ratings_df = pd.read_csv(rating_path)
        total_rows = len(ratings_df)
        print(f"[INFO] Total baris rating.csv: {total_rows:,}")
        if total_rows > sample_size:
            ratings_df = ratings_df.sample(n=sample_size, random_state=RANDOM_STATE)
            ratings_df = ratings_df.reset_index(drop=True)
            print(f"[INFO] Setelah sampling: {ratings_df.shape[0]:,} baris")
    else:
        ratings_df = pd.read_csv(rating_path)
        print(f"[INFO] rating.csv dimuat: {ratings_df.shape[0]:,} baris, {ratings_df.shape[1]} kolom")

    print(f"       Kolom: {list(ratings_df.columns)}")
    print()
    return movies_df, ratings_df


def preprocess_data(movies_df: pd.DataFrame, ratings_df: pd.DataFrame):
    print("=" * 60)
    print("STEP 2: Preprocessing Data")
    print("=" * 60)

    before = len(movies_df)
    movies_df = movies_df[
        (movies_df["genres"].notna()) &
        (movies_df["genres"].str.strip() != "") &
        (movies_df["genres"] != "(no genres listed)")
    ].copy()
    after = len(movies_df)
    print(f"[INFO] Hapus film tanpa genre: {before} -> {after} ({before - after} dihapus)")

    movies_df["genres_clean"] = movies_df["genres"].str.replace("|", " ", regex=False)
    print(f"[INFO] Format genre diubah (pipe -> spasi)")
    print(f"       Contoh: '{movies_df.iloc[0]['genres']}' -> '{movies_df.iloc[0]['genres_clean']}'")

    movies_df["movieId"] = movies_df["movieId"].astype(int)
    ratings_df["movieId"] = ratings_df["movieId"].astype(int)
    print(f"[INFO] Tipe data movieId: {movies_df['movieId'].dtype}")

    valid_movie_ids = set(movies_df["movieId"].unique())
    before_r = len(ratings_df)
    ratings_df = ratings_df[ratings_df["movieId"].isin(valid_movie_ids)].copy()
    after_r = len(ratings_df)
    print(f"[INFO] Hapus rating tanpa movie: {before_r:,} -> {after_r:,} ({before_r - after_r:,} dihapus)")

    movies_na = movies_df.isna().sum().sum()
    ratings_na = ratings_df.isna().sum().sum()
    print(f"[INFO] Missing values - movies: {movies_na}, ratings: {ratings_na}")
    if movies_na > 0:
        movies_df = movies_df.dropna().reset_index(drop=True)
    if ratings_na > 0:
        ratings_df = ratings_df.dropna().reset_index(drop=True)
    print(f"[INFO] Setelah penanganan missing value - movies: {len(movies_df)}, ratings: {len(ratings_df):,}")
    print()
    return movies_df, ratings_df


def split_data(ratings_df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42):
    print("=" * 60)
    print("STEP 3: Data Splitting")
    print("=" * 60)

    train_df, test_df = train_test_split(
        ratings_df,
        test_size=test_size,
        random_state=random_state
    )
    train_df = train_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    print(f"[INFO] Total rating: {len(ratings_df):,}")
    print(f"[INFO] Train set   : {len(train_df):,} ({(1 - test_size) * 100:.0f}%)")
    print(f"[INFO] Test set    : {len(test_df):,} ({test_size * 100:.0f}%)")
    print()
    return train_df, test_df


def build_tfidf(movies_df: pd.DataFrame):
    print("=" * 60)
    print("STEP 4: TF-IDF Embedding")
    print("=" * 60)

    vectorizer = TfidfVectorizer()

    tfidf_matrix = vectorizer.fit_transform(movies_df["genres_clean"])

    movie_indices = {
        movie_id: idx
        for idx, movie_id in enumerate(movies_df["movieId"].values)
    }

    print(f"[INFO] Jumlah film          : {tfidf_matrix.shape[0]}")
    print(f"[INFO] Jumlah fitur (genre) : {tfidf_matrix.shape[1]}")
    print(f"[INFO] Fitur genre          : {vectorizer.get_feature_names_out().tolist()}")
    print()
    return tfidf_matrix, vectorizer, movie_indices


def train_model(movies_df, train_df, like_threshold=4.0):
    print("=" * 60)
    print("STEP 5: Train Model")
    print("=" * 60)

    tfidf_matrix, vectorizer, movie_indices = build_tfidf(movies_df)

    train_movie_ids = set(train_df["movieId"].unique())
    tfidf_movie_ids = set(movie_indices.keys())
    covered = train_movie_ids.intersection(tfidf_movie_ids)
    missing = train_movie_ids - tfidf_movie_ids
    coverage = len(covered) / len(train_movie_ids) * 100 if len(train_movie_ids) > 0 else 0

    print(f"[INFO] Movie coverage     : {len(covered):,}/{len(train_movie_ids):,} ({coverage:.1f}%)")
    if len(missing) > 0:
        print(f"[WARNING] {len(missing)} movieId di train tidak ada di TF-IDF matrix")

    liked_ratings = train_df[train_df["rating"] >= like_threshold]
    users_with_likes = liked_ratings["userId"].nunique()
    total_users = train_df["userId"].nunique()
    print(f"[INFO] User dengan liked movies : {users_with_likes:,}/{total_users:,} ({users_with_likes/total_users*100:.1f}%)")
    print(f"[INFO] Total liked ratings       : {len(liked_ratings):,}")

    sample_user = liked_ratings["userId"].iloc[0]
    sample_profile, sample_liked = build_user_profile(
        sample_user, train_df, tfidf_matrix, movie_indices, like_threshold
    )
    if sample_profile is not None:
        sample_sim = cosine_similarity(sample_profile, tfidf_matrix)
        print(f"[INFO] Model validation OK  : sample user {sample_user} -> profile shape {sample_profile.shape}, sim range [{sample_sim.min():.4f}, {sample_sim.max():.4f}]")
    else:
        print(f"[WARNING] Model validation gagal untuk sample user {sample_user}")

    print(f"[INFO] Model telah di-train dan siap digunakan")
    print()
    return tfidf_matrix, vectorizer, movie_indices


def build_user_profile(user_id: int, train_df: pd.DataFrame,
                       tfidf_matrix, movie_indices: dict,
                       like_threshold: float = 4.0):

    user_ratings = train_df[train_df["userId"] == user_id]

    liked_ratings = user_ratings[user_ratings["rating"] >= like_threshold]
    liked_movie_ids = liked_ratings["movieId"].tolist()

    liked_movie_ids = [mid for mid in liked_movie_ids if mid in movie_indices]

    if len(liked_movie_ids) == 0:
        return None, []

    indices = [movie_indices[mid] for mid in liked_movie_ids]

    liked_vectors = tfidf_matrix[indices]
    user_profile = liked_vectors.mean(axis=0)
    
    user_profile = np.asarray(user_profile)
    if user_profile.ndim == 1:
        user_profile = user_profile.reshape(1, -1)

    return user_profile, liked_movie_ids


def get_recommendations(user_id: int, user_profile, tfidf_matrix,
                        movie_indices: dict, movies_df: pd.DataFrame,
                        train_df: pd.DataFrame, top_n: int = 10):

    similarity_scores = cosine_similarity(user_profile, tfidf_matrix).flatten()
    rated_movie_ids = set(train_df[train_df["userId"] == user_id]["movieId"].tolist())
    
    movie_ids = movies_df["movieId"].values
    scored_movies = []
    for i, movie_id in enumerate(movie_ids):
        if movie_id not in rated_movie_ids:
            scored_movies.append((movie_id, similarity_scores[i]))

    scored_movies.sort(key=lambda x: x[1], reverse=True)
    top_movies = scored_movies[:top_n]

    recommendations = []
    movie_lookup = movies_df.set_index("movieId")[["title", "genres"]].to_dict("index")

    for movie_id, score in top_movies:
        info = movie_lookup.get(movie_id, {})
        recommendations.append({
            "movieId": int(movie_id),
            "title": info.get("title", "Unknown"),
            "genres": info.get("genres", "").split("|"),
            "similarity_score": round(float(score), 4)
        })

    return recommendations


def evaluate(user_id: int, recommendations: list, test_df: pd.DataFrame,
             k: int, like_threshold: float = 4.0):

    user_test_all = test_df[test_df["userId"] == user_id]
    user_test_liked = user_test_all[user_test_all["rating"] >= like_threshold]
    relevant_items = set(user_test_liked["movieId"].tolist())
    total_relevant = len(relevant_items)

    recommended_items = set([rec["movieId"] for rec in recommendations[:k]])

    relevant_recommended = recommended_items.intersection(relevant_items)
    relevant_count = len(relevant_recommended)
    precision_at_k = relevant_count / k if k > 0 else 0.0
    recall_at_k = relevant_count / total_relevant if total_relevant > 0 else 0.0

    if total_relevant == 0:
        diagnostic = (f"User {user_id} tidak memiliki film yang disukai (rating >= {like_threshold}) "
                      f"di data test ({len(user_test_all)} total rating di test). "
                      f"Coba user lain yang memiliki lebih banyak rating.")
    elif relevant_count == 0:
        diagnostic = (f"User {user_id} punya {total_relevant} film relevan di test, "
                      f"tapi tidak ada yang cocok dengan top-{k} rekomendasi. "
                      f"Coba naikkan nilai K.")
    else:
        diagnostic = f"{relevant_count} dari {k} rekomendasi cocok dengan {total_relevant} film relevan di test."

    return {
        "precision_at_k": round(precision_at_k, 4),
        "recall_at_k": round(recall_at_k, 4),
        "relevant_count": relevant_count,
        "total_relevant": total_relevant,
        "total_test_ratings": len(user_test_all),
        "k": k,
        "diagnostic": diagnostic
    }


movies_df = None
ratings_df = None
train_df = None
test_df = None
tfidf_matrix = None
vectorizer = None
movie_indices = None
model_trained = False


def initialize_system():
    global movies_df, ratings_df, train_df, test_df
    global tfidf_matrix, vectorizer, movie_indices, model_trained

    start_time = time.time()
    base_dir = os.path.dirname(os.path.abspath(__file__))
    movie_path = os.path.join(base_dir, "movie.csv")
    rating_path = os.path.join(base_dir, "rating.csv")

    movies_df, ratings_df = load_dataset(movie_path, rating_path, RATING_SAMPLE_SIZE)
    movies_df, ratings_df = preprocess_data(movies_df, ratings_df)
    train_df, test_df = split_data(ratings_df, TEST_SIZE, RANDOM_STATE)

    tfidf_matrix, vectorizer, movie_indices = train_model(movies_df, train_df, LIKE_THRESHOLD)
    model_trained = True

    elapsed = time.time() - start_time
    print(f"[INFO] Sistem siap dalam {elapsed:.2f} detik")
    print(f"[INFO] Jumlah user unik di train: {train_df['userId'].nunique():,}")
    print(f"[INFO] Jumlah film: {len(movies_df):,}")
    print("=" * 60)
    print()


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/recommend", methods=["POST"])
def recommend():
    if not model_trained:
        return jsonify({
            "error": True,
            "message": "Model belum di-train. Restart server untuk menjalankan training."
        }), 503

    data = request.get_json()
    user_id = int(data.get("user_id", 1))
    top_n = int(data.get("top_n", 10))
    if user_id not in train_df["userId"].values:
        return jsonify({
            "error": True,
            "message": f"User ID {user_id} tidak ditemukan dalam dataset training. "
                       f"Coba User ID antara {train_df['userId'].min()} - {train_df['userId'].max()}."
        }), 404

    user_profile, liked_movie_ids = build_user_profile(
        user_id, train_df, tfidf_matrix, movie_indices, LIKE_THRESHOLD
    )

    if user_profile is None:
        return jsonify({
            "error": True,
            "message": f"User ID {user_id} tidak memiliki film yang disukai (rating >= {LIKE_THRESHOLD}) "
                       f"di data training."
        }), 404

    recommendations = get_recommendations(
        user_id, user_profile, tfidf_matrix,
        movie_indices, movies_df, train_df, top_n
    )
    metrics = evaluate(user_id, recommendations, test_df, top_n, LIKE_THRESHOLD)

    user_test_count = int(len(test_df[test_df["userId"] == user_id]))
    user_test_liked = int(len(test_df[
        (test_df["userId"] == user_id) & (test_df["rating"] >= LIKE_THRESHOLD)
    ]))
    
    # Extract liked genres
    liked_genres_str = movies_df[movies_df["movieId"].isin(liked_movie_ids)]["genres"].tolist()
    all_liked_genres = set()
    for genres_str in liked_genres_str:
        if isinstance(genres_str, str):
            all_liked_genres.update(genres_str.split("|"))
    liked_genres = sorted(list(all_liked_genres))
    
    user_info = {
        "user_id": user_id,
        "total_ratings_train": int(len(train_df[train_df["userId"] == user_id])),
        "liked_movies_count": len(liked_movie_ids),
        "total_ratings_test": user_test_count,
        "liked_movies_test": user_test_liked,
        "top_n": top_n,
        "liked_genres": liked_genres
    }

    return jsonify({
        "error": False,
        "recommendations": recommendations,
        "metrics": metrics,
        "user_info": user_info
    })


@app.route("/users", methods=["GET"])
def get_users():
    # Cari user yang punya liked movies di TRAIN (untuk profil)
    train_liked = train_df[train_df["rating"] >= LIKE_THRESHOLD]
    users_train_liked = set(train_liked["userId"].unique())

    # Cari user yang punya liked movies di TEST (untuk evaluasi)
    test_liked = test_df[test_df["rating"] >= LIKE_THRESHOLD]
    users_test_liked = set(test_liked["userId"].unique())

    # User yang evaluable: punya liked di train DAN test
    evaluable_users = sorted(users_train_liked.intersection(users_test_liked))

    # Ambil user dengan banyak test liked items agar metrik lebih meaningful
    test_liked_counts = test_liked.groupby("userId").size()
    top_evaluable = test_liked_counts[test_liked_counts.index.isin(evaluable_users)]
    top_evaluable = top_evaluable.sort_values(ascending=False)
    sample_users = top_evaluable.head(20).index.tolist()

    return jsonify({
        "sample_users": sample_users,
        "total_users": len(evaluable_users)
    })

if __name__ == "__main__":
    initialize_system()
    app.run(debug=False, host="127.0.0.1", port=5000)
