from fastapi import FastAPI, Query, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field
import logging
import time
from contextlib import asynccontextmanager
from datetime import datetime
import uvicorn
import pandas as pd
import ast
from fastapi.encoders import jsonable_encoder

from searcher import Searcher
from config import *
# Global searcher instance (loaded once)
searcher = None

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def cast_to_list(val):
    # Normalize many possible shapes into a Python list of strings
    if pd.isna(val) or val is None:
        return []
    if isinstance(val, list):
        return [str(x).strip() for x in val if x is not None]
    if isinstance(val, str):
        s = val.strip()
        # If the string looks like a Python/JSON list "['A','B']" or '["A","B"]'
        if (s.startswith('[') and s.endswith(']')):
            try:
                parsed = ast.literal_eval(s)  # safe for python list strings
                if isinstance(parsed, (list, tuple)):
                    return [str(x).strip() for x in parsed if x is not None]
            except Exception:
                pass
        # otherwise split by comma fallback
        return [p.strip() for p in s.split(',') if p.strip()]
    # fallback for other types
    return [str(val)]

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown logic."""
    global searcher
    logger.info("Starting Movie Search API...")
    start_time = time.time()

    try:
        searcher = Searcher()
        logger.info(f"Search engine loaded in {time.time() - start_time:.2f}s")
        # Before yield: startup code
        yield   # App runs here
        # After yield: shutdown code
    finally:
        logger.info("Shutting down Movie Search API...")

app = FastAPI(title="Movie Search API", description="API for searching movies using embeddings and FAISS index.", lifespan=lifespan)



class SearchRequest(BaseModel):
    query: str = Field(..., description= "Natural Language search query" , example="action movie with Tom cruise")
    top_n: int = Field(10, ge=1, le=50, description="Number of top results to return (1-50)")

class HealthResponse(BaseModel):
    status: str
    message: str
    timestamp: str
    system_info: dict



# API Routes
@app.get("/", response_class=HTMLResponse)
async def root():
    """Serve a simple web interface"""
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Movie Search API</title>
        <style>
            body { font-family: Arial, sans-serif; max-width: 900px; margin: 40px auto; padding: 20px; }
            h1 { color: #333; }
            .search-box { width: 100%; padding: 12px; font-size: 16px; margin: 20px 0;
                          border: 2px solid #ccc; border-radius: 8px; }
            .button { background: #007bff; color: #fff; padding: 12px 24px; border: none;
                      border-radius: 8px; cursor: pointer; font-size: 16px; }
            .button:hover { background: #0056b3; }
            .result { border: 1px solid #ddd; padding: 15px; margin: 12px 0;
                      border-radius: 8px; background: #fafafa; }
            .loading { color: #666; font-style: italic; }
            .timing { color: #888; font-size: 12px; margin-top: 10px; }
            .title { font-weight: bold; font-size: 18px; }
        </style>
    </head>
    <body>
        <h1>Movie Semantic Search</h1>
        <p>Search through 869K+ movies using natural language queries.</p>

        <input type="text" class="search-box" id="searchInput"
               placeholder="e.g. 'sci-fi thriller about time travel'">
        <button class="button" onclick="search()">Search Movies</button>

        <div id="results"></div>

        <h2>Example Queries</h2>
        <ul>
            <li><a href="#" onclick="searchExample('crime movie with Al Pacino and Robert De Niro')">
                crime movie with Al Pacino and Robert De Niro</a></li>
            <li><a href="#" onclick="searchExample('sci-fi thriller about time travel')">
                sci-fi thriller about time travel</a></li>
            <li><a href="#" onclick="searchExample('animated movie by Pixar')">
                animated movie by Pixar</a></li>
            <li><a href="#" onclick="searchExample('Marvel superhero movie in New York')">
                Marvel superhero movie in New York</a></li>
        </ul>

        <p><strong>API Documentation:</strong>
           <a href="/docs" target="_blank">Swagger UI</a> |
           <a href="/redoc" target="_blank">ReDoc</a></p>

        <script>
            function searchExample(query) {
                document.getElementById('searchInput').value = query;
                search();
            }

            async function search() {
                const query = document.getElementById('searchInput').value.trim();
                if (!query) return;

                const resultsDiv = document.getElementById('results');
                resultsDiv.innerHTML = '<div class="loading">Searching…</div>';

                try {
                    const response = await fetch('/search', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ query: query, top_n: 10 })
                    });

                    const data = await response.json();

                    let html = `<h3>Results for: "${query}"</h3>`;
                    html += `<div class="timing">Returned ${data.length} results</div>`;

                    data.forEach((movie, i) => {
                        const cast = movie.cast_crew
                            ? movie.cast_crew.slice(0, 5).join(', ')
                            : 'N/A';
                        html += `
                            <div class="result">
                                <div class="title">${i + 1}. ${movie.title || 'Untitled'}</div>
                                ${movie.release_date ? `<div><strong>Year:</strong> ${movie.release_date.split('-')[0]}</div>` : ''}
                                ${movie.genres ? `<div><strong>Genres:</strong> ${movie.genres}</div>` : ''}
                                <div><strong>Cast & Crew:</strong> ${cast}</div>
                                ${movie.popularity ? `<div><strong>Popularity:</strong> ${movie.popularity.toFixed(2)}</div>` : ''}
                                ${movie.score ? `<div><strong>Score:</strong> ${movie.score.toFixed(3)}</div>` : ''}
                                ${movie.overview ? `<div><strong>Overview:</strong> ${movie.overview.substring(0, 200)}…</div>` : ''}
                            </div>`;
                    });

                    resultsDiv.innerHTML = html;
                } catch (error) {
                    resultsDiv.innerHTML = `<div style="color: red;">Error: ${error.message}</div>`;
                }
            }

            document.getElementById('searchInput')
                .addEventListener('keypress', e => { if (e.key === 'Enter') search(); });
        </script>
    </body>
    </html>
    """
    return html_content

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    global searcher
    
    status = "healthy" if searcher is not None else "unhealthy"
    message = "Search engine is ready" if searcher is not None else "Search engine not loaded"
    
    return HealthResponse(
        status=status,
        message=message,
        timestamp=datetime.now().isoformat(),
        system_info={
            "model": MODEL,
            "device": DEVICE,
            "cross_encoder": CROSS_ENCODER,
            "index_metric": INDEX_METRIC
        }
    )


@app.post("/search")
async def search_movies(request: SearchRequest):
    """
    Search movies using natural language queries
    
    - **query**: Natural language search query (e.g., "action movie with explosions")
    - **top_n**: Number of results to return (1-50, default: 10)
    """
    global searcher
    
    if searcher is None:
        raise HTTPException(status_code=503, detail="Search engine not ready. Please wait for initialization.")
    
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")
    
    try:
        logger.info(f"Search query: '{request.query}' (top_n={request.top_n})")
        
        start_time = time.time()

        results_df = searcher.improved_search([request.query], top_n=request.top_n)

        search_time_ms = (time.time() - start_time) * 1000

        results_df = results_df.copy()
        if 'cast_crew' in results_df.columns:
            # Convert cast_crew strings to lists
            results_df['cast_crew'] = results_df['cast_crew'].apply(cast_to_list)
            # optionally ensure numeric types are native Python types
        
        results = jsonable_encoder(results_df.to_dict(orient='records'))

        logger.info(f"Search completed in {search_time_ms:.1f}ms, returned {len(results)} results")

        return results

    except Exception as e:
        logger.error(f"Search error: {e}")
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")
    
@app.get("/examples")
async def get_example_queries():
    """Get example queries for testing"""
    return {
        "examples": [
            "action movie with explosions",
            "romantic comedy from the 90s",
            "crime movie with Al Pacino and Robert De Niro",
            "sci-fi thriller about artificial intelligence",
            "animated movie by Pixar",
            "horror film with ghosts",
            "Marvel superhero movie in New York",
            "Christopher Nolan mind-bending thriller",
            "musical from Broadway",
            "western movie with Clint Eastwood"
        ]
    }

# CLI runner
if __name__ == "__main__":
    print("Starting Movie Search API...")
    print("Visit http://localhost:8000 for the web interface")
    print("Visit http://localhost:8000/docs for API documentation")
    
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )