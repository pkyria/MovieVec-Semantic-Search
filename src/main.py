from dataloader import DataLoader
from train import Embedder, FaissIndex
from searcher import Searcher
from config import *
import time


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

def main():
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

if __name__ == "__main__":
    main()
    



