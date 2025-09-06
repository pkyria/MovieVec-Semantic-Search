import faiss, numpy as np
from dataloader import DataLoader
from config import *
from train import Embedder, FaissIndex
from sentence_transformers import SentenceTransformer
import os.path

class Searcher:
    def __init__(self, index_file=INDEX_FILE):
        self.dataloader = DataLoader() # Initialize DataLoader
        self.data = self.dataloader.load_data() # Load data in list format using DataLoader
        self.embedder = Embedder(self.data) # Initialize embedder with data (to load model)
        self.model = self.embedder.model # Access the SentenceTransformer model
        
        self.faiss = None # Initialize FAISS index variable
        self.index = None # Initialize index variable
        # Initialize embedder and index
        if not os.path.exists(INDEX_FILE): # If index file does not exist, build the index
            print(f"Index file {INDEX_FILE} does not exist. Building the index...")
            self.build_index() # Build the index from data (This will also initialize FAISS index and save it to file)

         

    def build_index(self):
        """ Build the FAISS index from the dataset. """
        embeddings = self.embedder.encode(self.data)
        # Check if embeddings is not None and has correct shape
        if embeddings is not None: 
            dimension = embeddings.shape[1]
            print(f"Embedding dimension set to: {dimension}")
        else:
            print("Failed to set embedding dimension.")
            return
        print(f"Building FAISS index...")
        print(f"Index metric: {INDEX_METRIC}")
        print(f"Number of embeddings: {embeddings.shape[0]}")
        print(f"Embedding dimension: {dimension} - Type: {type(dimension)}")
        self.faiss = FaissIndex(dimension=dimension) # Initialize FAISS index with correct dimension
        self.faiss.add_embeddings(embeddings) # Add embeddings to the index
        self.faiss.save_index(INDEX_FILE) # Save the index to file
        self.index = self.faiss.load_index(INDEX_FILE) # Load pre-built index from file


    def search(self, query, top_k=8):
        """ Search for the top_k nearest neighbors of the query text. """
        query_embedding = self.model.encode(query, convert_to_numpy=True) # Encode the query text to get its embedding (Generate/load existing embedding)
        self.faiss_model = FaissIndex(dimension = query_embedding.shape[1]) # Initialize FAISS index with correct dimension
        self.index = self.faiss_model.load_index(INDEX_FILE) # Load pre-built index from file
        faiss.normalize_L2(query_embedding)  # Normalize query embedding for cosine similarity
        distances, indices = self.index.search(query_embedding, top_k)
        
        for i, q in enumerate(query):
            print(f"\n🔎 Query: {q}")
            print("Indices:", indices[i])
            print("Distances:", distances[i])

        metadata = self.dataloader.df # Get the full dataframe from DataLoader
        results = metadata.loc[metadata['index'].isin(indices[0])] # Retrieve metadata for the found indices
        return results