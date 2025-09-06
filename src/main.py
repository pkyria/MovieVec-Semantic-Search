from dataloader import DataLoader
from train import Embedder, FaissIndex
from searcher import Searcher
from config import *

query  = ['a sci-fi thriller with space travel and aliens']
results = Searcher().search(query)
print("\nSearch Results:\n")
i = 0
for idx, row in results.iterrows():
    print(f"{i+1}) Title: {row['title']}, Overview: {row['overview']}\n")  # Print first 100 chars of overview
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
    
    print(f"Cast & Crew: {cast_crew}\n\n")
        