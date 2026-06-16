from flask import Flask, render_template, request, session, jsonify, redirect, flash
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
from pymongo import MongoClient
from werkzeug.security import generate_password_hash, check_password_hash
from datetime import datetime
from scipy.sparse import hstack
import os

app = Flask(__name__)
app.secret_key = "cinevault_secret_2024"

# =====================================================
# MongoDB Connection
# =====================================================

client = MongoClient(os.environ.get("MONGO_URI"))

db = client["cinevault"]
users_collection      = db["users"]
history_collection    = db["history"]
watchlist_collection  = db["watchlist"]

# =====================================================
# Load Movie Dataset
# =====================================================

movies = pd.read_csv("data/movies.csv")

movies["genres"] = movies["genres"].fillna("")
movies["cast"]   = movies["cast"].fillna("")
movies["year"]   = movies["year"].fillna("").astype(str)

if "trending_score" not in movies.columns:
    movies["trending_score"] = 0

movies["trending_score"] = pd.to_numeric(
    movies["trending_score"], errors="coerce"
).fillna(0)

# Reset index to ensure clean 0..n-1 positional indices
movies = movies.reset_index(drop=True)

movies["genres"] = movies["genres"].fillna("")
movies["cast"]   = movies["cast"].fillna("")

# Repeat genres to weight them heavier than cast
movies["features"] = (
    (movies["genres"] + " ") * 3 + movies["cast"]
)

# =====================================================
# Recommendation Engine
# =====================================================


genre_tfidf = TfidfVectorizer(token_pattern=r"[^|]+")  # split on the | delimiter
cast_tfidf  = TfidfVectorizer(token_pattern=r"[^|]+")

genre_matrix = genre_tfidf.fit_transform(movies["genres"])
cast_matrix  = cast_tfidf.fit_transform(movies["cast"])



# Weight genres higher (e.g. 2x) since they define the "type" of film
tfidf_matrix = hstack([genre_matrix * 2.0, cast_matrix * 1.0]).tocsr()

knn_model = NearestNeighbors(metric="cosine", algorithm="brute")
knn_model.fit(tfidf_matrix)


def find_movie_index(title):
    """
    Find the best matching index for a title query.
    Tries exact (case-insensitive) match first, then falls back
    to substring match. Returns None if nothing matches.
    """
    if not title:
        return None

    # Exact match (case-insensitive) — most reliable
    exact = movies[movies["title"].str.lower() == title.lower().strip()]
    if not exact.empty:
        return exact.index[0]

    # Substring match — regex=False avoids errors with special chars
    # like ., (, ), +, * which are common in movie titles
    partial = movies[
        movies["title"].str.contains(title, case=False, na=False, regex=False)
    ]
    if not partial.empty:
        return partial.index[0]

    return None


def recommend(title, n_recommendations=8):
    idx = find_movie_index(title)

    if idx is None:
        return []

    movie_vector = tfidf_matrix[idx]

    n_neighbors = min(n_recommendations + 1, len(movies))

    distances, indices = knn_model.kneighbors(
        movie_vector,
        n_neighbors=n_neighbors
    )

    neighbor_indices = [i for i in indices.flatten() if i != idx][:n_recommendations]

    return movies.iloc[neighbor_indices].to_dict(orient="records")


# =====================================================
# Search History (per-user, MongoDB)
# =====================================================

from datetime import datetime, timezone
from zoneinfo import ZoneInfo

def add_to_search_history(query, search_type="title"):
    """Record a search query in per-user search history."""
    if "user" not in session:
        return

    username = session["user"]

    # Remove duplicate entry if it exists
    history_collection.delete_one({
        "username": username,
        "query": query,
        "search_type": search_type
    })

    # Store time in UTC
    history_collection.insert_one({
        "username": username,
        "query": query,
        "search_type": search_type,
        "searched_at": datetime.now(timezone.utc)
    })

    # Keep only latest 50 searches
    docs = list(
        history_collection.find(
            {"username": username},
            sort=[("searched_at", -1)]
        )
    )

    if len(docs) > 50:
        old_ids = [d["_id"] for d in docs[50:]]
        history_collection.delete_many({
            "_id": {"$in": old_ids}
        })


def get_search_history():
    """Return search history with IST timestamps."""
    if "user" not in session:
        return []

    username = session["user"]

    docs = history_collection.find(
        {"username": username},
        sort=[("searched_at", -1)]
    )

    result = []

    for doc in docs:
        utc_time = doc.get("searched_at")

        if utc_time:
            # Handle both timezone-aware and naive datetimes
            if utc_time.tzinfo is None:
                utc_time = utc_time.replace(tzinfo=timezone.utc)

            ist_time = utc_time.astimezone(
                ZoneInfo("Asia/Kolkata")
            )

            formatted_time = ist_time.strftime(
                "%b %d, %Y · %I:%M %p"
            )
        else:
            formatted_time = "Unknown"

        result.append({
            "query": doc.get("query", ""),
            "search_type": doc.get("search_type", "title"),
            "searched_at": formatted_time
        })

    return result


# =====================================================
# Watchlist (per-user, MongoDB)
# =====================================================

def get_user_watchlist():
    if "user" not in session:
        return []

    username = session["user"]
    docs = watchlist_collection.find(
        {"username": username},
        sort=[("added_at", -1)]
    )

    results = []
    for doc in docs:
        match = movies[movies["title"] == doc["title"]]
        if not match.empty:
            results.append(match.iloc[0].to_dict())
    return results


def add_to_watchlist(title):
    if "user" not in session:
        return {"status": "error", "message": "Not logged in"}

    username = session["user"]
    exists   = watchlist_collection.find_one({"username": username, "title": title})

    if exists:
        return {"status": "exists"}

    watchlist_collection.insert_one({
        "username": username,
        "title":    title,
        "added_at": datetime.utcnow()
    })
    return {"status": "added"}


def remove_from_watchlist(title):
    if "user" not in session:
        return {"status": "error"}

    username = session["user"]
    watchlist_collection.delete_one({"username": username, "title": title})
    return {"status": "removed"}


# =====================================================
# Search Function
# =====================================================

def search_movies(search_type, query=""):
    if search_type == "title":
        results = movies[movies["title"].str.contains(query, case=False, na=False, regex=False)]

    elif search_type == "genre":
        results = movies[movies["genres"].str.contains(query, case=False, na=False, regex=False)]

    elif search_type == "cast":
        results = movies[movies["cast"].str.contains(query, case=False, na=False, regex=False)]

    else:
        results = pd.DataFrame()

    return results.to_dict(orient="records")


# =====================================================
# Authentication Routes
# =====================================================

@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        username = request.form["username"].strip()
        email    = request.form["email"].strip()
        password = request.form["password"]
        confirm  = request.form["confirm"]

        if password != confirm:
            flash("Passwords do not match", "error")
            return redirect("/register")

        if users_collection.find_one({"username": username}):
            flash("Username already exists", "error")
            return redirect("/register")

        users_collection.insert_one({
            "username": username,
            "email":    email,
            "password": generate_password_hash(password)
        })

        flash("Account created successfully!", "success")
        return redirect("/login")

    return render_template("register.html")


@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form["username"]
        password = request.form["password"]
        user     = users_collection.find_one({"username": username})

        if user and check_password_hash(user["password"], password):
            session["user"] = username
            flash("Login successful!", "success")
            return redirect("/")

        flash("Invalid username or password", "error")

    return render_template("login.html")


@app.route("/logout")
def logout():
    session.clear()
    flash("Logged out successfully", "success")
    return redirect("/login")


# =====================================================
# Main Routes
# =====================================================

@app.route("/", methods=["GET", "POST"])
def index():
    if "user" not in session:
        return redirect("/login")

    # ── Handle GET search (coming from history links) ──────────────
    if request.method == "GET":
        search_type = request.args.get("search_type", "").strip()
        query       = request.args.get("query", "").strip()

        if search_type and query:
            add_to_search_history(query, search_type)

            if search_type == "title":
                recs = recommend(query)
                movie_data_list = movies[
                    movies["title"].str.contains(query, case=False, na=False, regex=False)
                ].to_dict(orient="records")

                if not movie_data_list:
                    return render_template("results.html", query=query, results=[])

                idx = find_movie_index(query)
                movie_data = movies.loc[idx].to_dict() if idx is not None else movie_data_list[0]

                return render_template(
                    "recommend.html",
                    movie=movie_data,
                    recommendations=recs
                )
            else:
                results = search_movies(search_type, query)
                return render_template("results.html", query=query, results=results)

    # ── Handle POST search (from the search form) ──────────────────
    if request.method == "POST":
        search_type = request.form.get("search_type", "title")
        query       = request.form.get("query", "").strip()

        if query:
            add_to_search_history(query, search_type)

        if search_type == "title" and query:
            recs = recommend(query)

            idx = find_movie_index(query)

            if idx is None:
                return render_template("results.html", query=query, results=[])

            movie_data = movies.loc[idx].to_dict()

            return render_template(
                "recommend.html",
                movie=movie_data,
                recommendations=recs
            )

        else:
            results = search_movies(search_type, query)
            return render_template(
                "results.html",
                query=query,
                results=results
            )

    # ── Default: show trending ─────────────────────────────────────
    trending = movies.sort_values(
        "trending_score", ascending=False
    ).to_dict(orient="records")

    return render_template(
        "index.html",
        movies=trending,
        username=session["user"]
    )


@app.route("/history")
def history_page():
    if "user" not in session:
        return redirect("/login")
    return render_template("history.html", username=session["user"])


@app.route("/movie/<title>")
def movie_detail(title):
    if "user" not in session:
        return redirect("/login")

    idx = find_movie_index(title)

    if idx is None:
        return render_template("results.html", query=title, results=[])

    movie_data = movies.loc[idx].to_dict()
    recs = recommend(title)

    return render_template(
        "recommend.html",
        movie=movie_data,
        recommendations=recs
    )

# =====================================================
# APIs
# =====================================================

@app.route("/api/history")
def api_history():
    if "user" not in session:
        return jsonify([])
    return jsonify(get_search_history())

@app.route("/api/history/remove", methods=["POST"])
def api_history_remove():
    if "user" not in session:
        return jsonify({"status": "error"}), 401

    data        = request.get_json(silent=True) or {}
    query       = data.get("query", "").strip()
    search_type = data.get("search_type", "title")

    history_collection.delete_one({
        "username":    session["user"],
        "query":       query,
        "search_type": search_type
    })
    return jsonify({"status": "removed"})

@app.route("/api/history/clear", methods=["POST"])
def api_history_clear():
    if "user" not in session:
        return jsonify({"status": "error"}), 401

    history_collection.delete_many({"username": session["user"]})
    return jsonify({"status": "cleared"})

@app.route("/api/watchlist")
def api_watchlist():
    if "user" not in session:
        return jsonify([])
    return jsonify(get_user_watchlist())

@app.route("/api/watchlist/add", methods=["POST"])
def api_watchlist_add():
    if "user" not in session:
        return jsonify({"status": "error", "message": "Not logged in"}), 401

    data  = request.get_json(silent=True) or {}
    title = data.get("title", "").strip()

    if not title:
        return jsonify({"status": "error", "message": "No title provided"}), 400

    return jsonify(add_to_watchlist(title))

@app.route("/api/watchlist/remove", methods=["POST"])
def api_watchlist_remove():
    if "user" not in session:
        return jsonify({"status": "error", "message": "Not logged in"}), 401

    data  = request.get_json(silent=True) or {}
    title = data.get("title", "").strip()

    if not title:
        return jsonify({"status": "error", "message": "No title provided"}), 400

    return jsonify(remove_from_watchlist(title))

@app.route("/api/watchlist/status")
def api_watchlist_status():
    if "user" not in session:
        return jsonify({"in_watchlist": False})

    title  = request.args.get("title", "").strip()
    exists = watchlist_collection.find_one({
        "username": session["user"],
        "title":    title
    })
    return jsonify({"in_watchlist": bool(exists)})

@app.route("/api/trending")
def api_trending():
    top = movies.nlargest(10, "trending_score").to_dict(orient="records")
    return jsonify(top)

# =====================================================
# Run App
# =====================================================

if __name__ == "__main__":
    app.run(debug=True)
