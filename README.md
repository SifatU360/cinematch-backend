# CineMatch — Backend

## Setup

```bash
pip install -r requirements.txt
```

Create `backend/.env` (already added):

```bash
BACKEND_CORS_ORIGINS=http://localhost:5173
BACKEND_HOST=127.0.0.1
BACKEND_PORT=8000
```

Copy your 4 pickle files into this folder:
```
backend/
  tfidf_matrix.pkl
  tfidf.pkl
  df.pkl
  indices.pkl
  main.py
  requirements.txt
```

## Run

```bash
uvicorn main:app --reload --port 8000
```

API docs → http://localhost:8000/docs

## Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/api/recommend?title=Toy Story&n=10` | Get recommendations |
| GET | `/api/search?q=avenger&limit=10` | Autocomplete search |
| GET | `/api/movie/{title}` | Movie details |
| GET | `/api/popular?limit=12` | Top popular movies |
| GET | `/api/stats` | Dataset statistics |
