from dataloader import DataLoader
from train import Embedder, FaissIndex
from searcher import Searcher
from config import *
import time

query  = ['crime movie with al pacino and robert de niro']  # Example query

# timings = Searcher().diagnose_performance(query)

# print(timings)


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
    
    print(f"Found {len(results)} results in {elapsed:.3f}s")
        