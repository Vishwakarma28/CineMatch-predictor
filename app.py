from flask import Flask, request, jsonify
import joblib
import pandas as pd
import requests
from flask_cors import CORS
from sklearn.metrics.pairwise import cosine_similarity
from urllib.parse import quote
import requests
from dotenv import load_dotenv
import os

app = Flask(__name__)
CORS(app)

app.config['TMDB_API_KEY'] = os.getenv("TMDB_API_KEY")

# Load saved objects
indices = joblib.load('indices.pkl')
tfidf_matrix = joblib.load('tfidf_matrix.pkl')
tfidf = joblib.load('tfidf.pkl')
df = pd.read_pickle('df.pkl')


# Recommendation function
def recommend(title, n=10):

    if title not in indices:
        return ['Movie not found']

    idx = indices[title]

    sim_score = cosine_similarity(tfidf_matrix[idx], tfidf_matrix).flatten()

    similar_idx = sim_score.argsort()[::-1][1:n+1]

    return df['title'].iloc[similar_idx].tolist()

def get_movie_details(movie_title):

    api_key = os.getenv("TMDB_API_KEY") 
    movie_title = movie_title.replace("  ", " ").strip()

    try:
        url = f"https://api.themoviedb.org/3/search/movie?api_key={api_key}&query={quote(movie_title)}"

        response = requests.get(url, timeout=5)
        data = response.json()
        
        results = data.get("results")
        # print("responce==================>",results)
        if results:
            movie = results[0]

            poster_path = movie.get("poster_path")
            backdrop_path = movie.get("backdrop_path")

            return {
                "title": movie.get("title"),
                "overview": movie.get("overview"),
                "rating": movie.get("vote_average"),
                "release_date": movie.get("release_date"),
                "poster": f"https://image.tmdb.org/t/p/w500{poster_path}" if poster_path else "https://via.placeholder.com/500x750.png?text=No+Poster",
                "backdrop": f"https://image.tmdb.org/t/p/original{backdrop_path}" if backdrop_path else None
            }

    except Exception as e:
        print("Movie fetch error:", e)

    return {
        "title": movie_title,
        "overview": "No description available",
        "rating": "N/A",
        "release_date": "N/A",
        "poster": "https://via.placeholder.com/500x750.png?text=No+Poster",
        "backdrop": None
    }


def get_movie_poster(movie_title):

    api_key = os.getenv("TMDB_API_KEY") 
    movie_title = movie_title.replace("  ", " ").strip()
    

    try:
        url = f"https://api.themoviedb.org/3/search/movie?api_key={api_key}&query={quote(movie_title)}"

        response = requests.get(url, timeout=5)
        data = response.json()

        results = data.get("results", [])

        if results:
            poster_path = results[0].get("poster_path")

            if poster_path:
                return f"https://image.tmdb.org/t/p/w500{poster_path}"

    except Exception as e:
        print("Poster fetch error:", e)

    return "https://via.placeholder.com/500x750.png?text=No+Poster"


@app.route("/")
def home():
    return "Movie Recommendation API is running 🚀"


@app.route('/recommend', methods=['POST'])
def predict():

    try:
        data = request.get_json()

        title = data.get("movie") or data.get("title")

        if not title:
            return jsonify({"error": "Movie title is required"}), 400

        # Get full details for searched movie
        movie_details = get_movie_details(title)

        # Get recommendations from model
        recommendations = recommend(title)
        print("reco", recommendations)

        result = []

        for movie in recommendations:
            print("movie", movie)

            poster = get_movie_poster(movie)

            result.append({
                "title": movie,
                "poster": poster
            })

        return jsonify({
            "movie": movie_details,   # full details only for searched movie
            "recommendations": result
        })

    except Exception as e:
        return jsonify({
            "error": str(e)
        }), 500


if __name__ == "__main__":
    app.run(debug=True)