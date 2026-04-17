import pickle
import os
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
from dotenv import load_dotenv
import warnings
warnings.filterwarnings("ignore")

load_dotenv()

cors_origins = [origin.strip() for origin in os.getenv("BACKEND_CORS_ORIGINS", "http://localhost:5173").split(",") if origin.strip()]

app = FastAPI(
    title="CineMatch API",
    description="AI-powered movie recommendation engine",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Load model artifacts ──────────────────────────────────────────────────────
print("Loading model artifacts...")
df           = pd.read_pickle("df.pkl")
indices      = pickle.load(open("indices.pkl", "rb"))
tfidf_matrix = pickle.load(open("tfidf_matrix.pkl", "rb"))
tfidf        = pickle.load(open("tfidf.pkl", "rb"))

# ── Clean dataframe on load ───────────────────────────────────────────────────
df["title"]        = df["title"].fillna("").astype(str)
df["overview"]     = df["overview"].fillna("").astype(str)
df["genres"]       = df["genres"].fillna("").astype(str)
df["tagline"]      = df["tagline"].fillna("").astype(str) if "tagline" in df.columns else ""
df["vote_average"] = pd.to_numeric(df["vote_average"], errors="coerce").fillna(0.0)
df["popularity"]   = pd.to_numeric(df["popularity"],   errors="coerce").fillna(0.0)

print(f"Loaded {len(df)} movies ✓")


# ── Schemas ───────────────────────────────────────────────────────────────────
class Movie(BaseModel):
    title: str
    overview: Optional[str] = ""
    genres: Optional[str] = ""
    tagline: Optional[str] = ""
    vote_average: Optional[float] = 0.0
    popularity: Optional[float] = 0.0
    similarity: Optional[float] = None


class RecommendResponse(BaseModel):
    query: str
    recommendations: List[Movie]
    total: int


# ── Helpers ───────────────────────────────────────────────────────────────────
def safe_float(val):
    try:
        f = float(val)
        return 0.0 if np.isnan(f) or np.isinf(f) else f
    except Exception:
        return 0.0


def row_to_dict(row, similarity=None):
    return {
        "title":        str(row["title"]),
        "overview":     str(row["overview"]),
        "genres":       str(row["genres"]),
        "tagline":      str(row["tagline"]) if "tagline" in row else "",
        "vote_average": safe_float(row["vote_average"]),
        "popularity":   safe_float(row["popularity"]),
        "similarity":   round(safe_float(similarity), 4) if similarity is not None else None,
    }


def get_recommendations(title: str, n: int = 10) -> List[dict]:
    if title not in indices:
        return []
    idx = indices[title]
    sim_scores = cosine_similarity(tfidf_matrix[idx], tfidf_matrix).flatten()
    top_indices = np.argsort(sim_scores)[::-1][1:n + 1]
    return [row_to_dict(df.iloc[i], sim_scores[i]) for i in top_indices]


# ── Routes ────────────────────────────────────────────────────────────────────
@app.get("/")
def root():
    return {"message": "CineMatch API is running", "movies": len(df)}


@app.get("/api/recommend", response_model=RecommendResponse)
def recommend(
    title: str = Query(..., description="Movie title to base recommendations on"),
    n:     int = Query(10, ge=1, le=50, description="Number of recommendations"),
):
    if title not in indices:
        raise HTTPException(status_code=404, detail=f"Movie '{title}' not found in database.")
    recs = get_recommendations(title, n)
    return RecommendResponse(query=title, recommendations=recs, total=len(recs))


@app.get("/api/search")
def search(q: str = Query(..., min_length=1), limit: int = Query(10, ge=1, le=50)):
    """Fuzzy title search for autocomplete."""
    q_lower = q.lower()
    matches = df[df["title"].str.lower().str.contains(q_lower, na=False)]["title"].tolist()
    return {"results": matches[:limit], "total": len(matches)}


@app.get("/api/movie/{title}")
def get_movie(title: str):
    """Get details for a single movie."""
    if title not in indices:
        raise HTTPException(status_code=404, detail=f"Movie '{title}' not found.")
    idx = indices[title]
    return row_to_dict(df.iloc[idx])


@app.get("/api/popular")
def popular(limit: int = Query(12, ge=1, le=50)):
    """Return top movies by popularity score."""
    top = df.nlargest(limit, "popularity")
    return {"results": [row_to_dict(row) for _, row in top.iterrows()]}


@app.get("/api/movies")
def all_movies(
    page:     int = Query(1, ge=1, description="Page number"),
    per_page: int = Query(15, ge=1, le=100, description="Items per page"),
    sort_by:  str = Query("popularity", description="Sort field: popularity | vote_average | title"),
    order:    str = Query("desc", description="asc or desc"),
    genre:    str = Query(None, description="Filter by genre word"),
    q:        str = Query(None, description="Title search filter"),
):
    """Paginated, sortable, filterable movie browser."""
    result = df.copy()

    # Filter by search
    if q:
        result = result[result["title"].str.lower().str.contains(q.lower(), na=False)]

    # Filter by genre
    if genre:
        result = result[result["genres"].str.lower().str.contains(genre.lower(), na=False)]

    # Sort
    sort_col = sort_by if sort_by in ["popularity", "vote_average", "title"] else "popularity"
    ascending = order == "asc"
    result = result.sort_values(sort_col, ascending=ascending)

    total = len(result)
    total_pages = max(1, (total + per_page - 1) // per_page)
    start = (page - 1) * per_page
    end   = start + per_page
    page_data = result.iloc[start:end]

    return {
        "results":     [row_to_dict(row) for _, row in page_data.iterrows()],
        "total":       total,
        "page":        page,
        "per_page":    per_page,
        "total_pages": total_pages,
    }


@app.get("/api/stats")
def stats():
    """Dataset statistics."""
    return {
        "total_movies":  len(df),
        "avg_rating":    round(float(df["vote_average"].mean()), 2),
        "total_genres":  len(set(" ".join(df["genres"].dropna()).split())),
    }
