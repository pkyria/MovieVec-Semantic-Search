from dataloader import DataLoader
from train import Embedder, FaissIndex
from searcher import Searcher
from config import *
import time
import argparse



def print_results(results):
    print("\nSearch Results:\n")
    i = 0
    for idx, row in results.iterrows():
        print(f"{i+1}) Title: {row['title']}, Overview: {row['overview']}\n") 
        cast_crew = ''
        if row['cast_crew']:
            cnt = 0
            for cast_member in row['cast_crew'].split(', ')[:5]:  # Print up to first 5 cast members
                cast_crew += cast_member + ', '
            if cast_crew.endswith(', '):
                cast_crew = cast_crew[:-2]
                
        else:
            cast_crew = 'N/A'
        i += 1
        
        print(f'"Genres: {row["genres"]}\n')
        print(f"Cast & Crew: {cast_crew}\n")
        print(f"Popularity: {row['popularity']}")
        print("-----\n")

def cli_mode():
    # Initialize once
    print("Initializing search engine...")
    searcher = Searcher()
    print("Search engine ready!")

    # Interactive loop 
    while True:
        query = [input("Search: ")] # Input query as a list 
        # Exit on 'quit'
        if query[0].lower() == 'quit': 
            break 
        # Time the search
        start = time.time()
        results = searcher.improved_search(query)
        elapsed = time.time() - start

        print_results(results)
        
        print(f"Found {len(results)} results in {elapsed:.3f}s")

def api_mode():
    """Start FastAPI server"""
    import uvicorn
    
    print("Starting Movie Search API...")
    print("Web interface: http://localhost:8000")
    print("API docs: http://localhost:8000/docs")
    print("\nPress Ctrl+C to stop the server\n")
    
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        reload=False,  # Set to True for development
        log_level="info"
    )


def main():
    """Main entry point with argument parsing"""
    parser = argparse.ArgumentParser(
        description="🎬 Movie Semantic Search Engine",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py              # Interactive CLI mode
  python main.py --api        # Start FastAPI server
  python main.py --cli        # Explicit CLI mode
        """
    )
    
    parser.add_argument(
        "--api", 
        action="store_true", 
        help="Start FastAPI web server"
    )
    
    parser.add_argument(
        "--cli", 
        action="store_true", 
        help="Start interactive CLI mode (default)"
    )
    
    parser.add_argument(
        "--port", 
        type=int, 
        default=8000, 
        help="Port for API server (default: 8000)"
    )
    
    args = parser.parse_args()
    
    # Determine mode
    if args.api:
        # Modify uvicorn call to use custom port
        import uvicorn
        print(f"Starting Movie Search API on port {args.port}...")
        uvicorn.run("app:app", host="0.0.0.0", port=args.port, reload=False)
    else:
        # Default to CLI mode
        cli_mode()

if __name__ == "__main__":
    main()
    


