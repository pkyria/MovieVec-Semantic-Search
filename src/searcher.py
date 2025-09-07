import faiss, numpy as np
from dataloader import DataLoader
from config import *
from train import Embedder, FaissIndex
from sentence_transformers import SentenceTransformer, CrossEncoder
import os.path
from sklearn.preprocessing import MinMaxScaler

class Searcher:
    def __init__(self, index_file=INDEX_FILE):
        self.dataloader = DataLoader() # Initialize DataLoader
        self.data = self.dataloader.load_data() # Load data in list format using DataLoader
        self.embedder = Embedder(self.data) # Initialize embedder with data (to load model)
        self.model = self.embedder.model # Access the SentenceTransformer model
        self.ids = None
        
        self.faiss = None # Initialize FAISS index variable
        self.index = None # Initialize index variable
        # Initialize embedder and index
        if not os.path.exists(INDEX_FILE): # If index file does not exist, build the index
            print(f"Index file {INDEX_FILE} does not exist. Building the index...")
            self.build_index() # Build the index from data (This will also initialize FAISS index and save it to file)

         

    def build_index(self):
        """ Build the FAISS index from the dataset. """
        embeddings, self.ids = self.embedder.encode(self.data)
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


    def search(self, query, top_k=200):
        """ Search for the top_k nearest neighbors of the query text. """
        query_embedding = self.model.encode(
                    query,
                    batch_size=BATCH_SIZE,
                    show_progress_bar=True,
                    device=DEVICE,
                    convert_to_numpy=True  # Convert to numpy immediately to save GPU memory (moves to CPU)
                ) # Encode the query text to get its embedding (Generate/load existing embedding)
        
        self.faiss_model = FaissIndex(dimension = query_embedding.shape[1]) # Initialize FAISS index with correct dimension
        self.index = self.faiss_model.load_index(INDEX_FILE) # Load pre-built index from file
        faiss.normalize_L2(query_embedding)  # Normalize query embedding for cosine similarity

        try:
            # Move index to GPU
            res = faiss.StandardGpuResources()  # manages GPU resources
            gpu_index = faiss.index_cpu_to_gpu(res, 0, self.index)  # 0 = GPU id
            distances, indices = gpu_index.search(query_embedding, top_k)
        except Exception as e:
            print(f"Error moving index to GPU: {e}. Falling back to CPU search.")
            distances, indices = self.index.search(query_embedding, top_k)

        
        
        self.ids = self.embedder.ids # Load the ids from the embedder
        
        for i, q in enumerate(query):
            print(f"\n🔎 Query: {q}")
            print("Indices:", indices[i])
            print("Distances:", distances[i])

        # Translate FAISS indices back to dataframe indices
        faiss_results = indices[0]
        df_indices = [self.ids[idx] for idx in faiss_results]

        metadata = self.dataloader.df # Get the full dataframe from DataLoader
        # results = metadata.loc[metadata['index'].isin(indices[0])] # Retrieve metadata for the found indices

        # Rerank results based on similarity and popularity
        results = self.rerank_results(metadata.iloc[df_indices], distances[0]).head(10)  # Return top 10 results after reranking
        # results = metadata.iloc[df_indices].head(10)  # Return all results without reranking for now

        return results
    



    
    def improved_search(self, query, top_k = 100):
        """ Search for the top_k nearest neighbors of the query text with cross-encoder re-ranking. """
        
        # Initialize cross-encoder if not already done (add this to your __init__ method ideally)
        if not hasattr(self, 'cross_encoder'):
            from sentence_transformers import CrossEncoder
            self.cross_encoder = CrossEncoder(CROSS_ENCODER)
            print("Cross-encoder loaded for re-ranking")
        
        query_embedding = self.model.encode(
            query,
            batch_size=BATCH_SIZE,
            show_progress_bar=True,
            device=DEVICE,
            convert_to_numpy=True # Convert to numpy immediately to save GPU memory (moves to CPU)
        ) # Encode the query text to get its embedding (Generate/load existing embedding)
        
        self.faiss_model = FaissIndex(dimension = query_embedding.shape[1]) # Initialize FAISS index with correct dimension
        self.index = self.faiss_model.load_index(INDEX_FILE) # Load pre-built index from file
        faiss.normalize_L2(query_embedding) # Normalize query embedding for cosine similarity
        self.ids = self.embedder.ids # Load the ids from the embedder
        # STAGE 1: Bi-encoder retrieval (get more candidates for re-ranking)
        retrieval_k = min(top_k * 3, len(self.ids))  # Get 3x more candidates for re-ranking
        
        try:
            # Move index to GPU
            res = faiss.StandardGpuResources() # manages GPU resources
            gpu_index = faiss.index_cpu_to_gpu(res, 0, self.index) # 0 = GPU id
            distances, indices = gpu_index.search(query_embedding, retrieval_k)
        except Exception as e:
            print(f"Error moving index to GPU: {e}. Falling back to CPU search.")
            distances, indices = self.index.search(query_embedding, retrieval_k)
        
        self.ids = self.embedder.ids # Load the ids from the embedder
        
        for i, q in enumerate(query):
            print(f"\n🔎 Query: {q}")
            print("Indices:", indices[i][:10])  # Show only first 10 for readability
            print("Distances:", distances[i][:10])
        
        # Translate FAISS indices back to dataframe indices
        faiss_results = indices[0]
        df_indices = [self.ids[idx] for idx in faiss_results]
        metadata = self.dataloader.df # Get the full dataframe from DataLoader
        
        # Get candidate results for re-ranking
        candidates_df = metadata.iloc[df_indices]
        
        # STAGE 2: Cross-encoder re-ranking
        print(f"Re-ranking {len(candidates_df)} candidates with cross-encoder...")
        
        # Prepare query-document pairs for cross-encoder
        query_text = query[0] if isinstance(query, list) else query  # Handle both string and list input
        
        # Create text representations of your movies (adjust field names as needed)
        candidate_texts = []
        for _, row in candidates_df.iterrows():
            # Combine relevant fields - adjust these field names to match your dataframe
            movie_text = f"{row.get('title', '')} {row.get('overview', '')} {row.get('genres', '')}"
            candidate_texts.append(movie_text.strip())
        
        # Get cross-encoder scores
        query_doc_pairs = [(query_text, doc_text) for doc_text in candidate_texts]
        cross_scores = self.cross_encoder.predict(query_doc_pairs)
        
        # Combine bi-encoder and cross-encoder scores
        combined_scores = []
        for i, (bi_distance, cross_score) in enumerate(zip(distances[0], cross_scores)):
            # Convert FAISS distance to similarity (assuming IndexFlatIP)
            bi_similarity = float(bi_distance)
            
            # Combine scores (you can adjust these weights)
            combined_score = bi_similarity * 0.3 + float(cross_score) * 0.7
            combined_scores.append(combined_score)
        
        # Add combined scores to the dataframe
        candidates_df = candidates_df.copy()  # Avoid SettingWithCopyWarning
        candidates_df.loc[:, 'combined_score'] = combined_scores
        candidates_df.loc[:, 'bi_score'] = distances[0].astype(float)
        candidates_df.loc[:, 'cross_score'] = cross_scores.astype(float)
        
        # Sort by combined score (descending)
        reranked_df = candidates_df.sort_values('combined_score', ascending=False)
        
        # Apply your existing rerank_results method on top 50 candidates (to avoid overwhelming it)
        top_candidates = reranked_df.head(50)
        final_distances = top_candidates['combined_score'].values
        
        print(f"Cross-encoder re-ranking completed. Top score: {final_distances[0]:.4f}")
        
        # Use your existing reranking method
        results = self.rerank_results(top_candidates, final_distances).head(10) # Return top 10 results after reranking
        
        return results

    def rerank_results(self, results_df, similarities, alpha=0.1, beta=0.2):
        """
        results_df: dataframe slice of FAISS results
        distances: array of FAISS distances for these results
        alpha: weight for similarity vs popularity (0.0 → only popularity, 1.0 → only similarity)
        """
        
        # Convert distances → similarities (invert)
        similarities = 1 - similarities  # higher = better

        # Normalize similarities per query
        norm_sims = (similarities - similarities.min()) / (similarities.max() - similarities.min())
        norm_pop = results_df["popularity_norm"].to_numpy()
        norm_cast = results_df["cast_crew_norm"].to_numpy()
        
        # Weighted score
        scores = alpha * norm_sims + beta * norm_cast + (1 - alpha - beta) * norm_pop
        
        # Attach scores to dataframe and sort
        results_df = results_df.copy()
        results_df["score"] = scores
        results_df = results_df.sort_values("score", ascending=False)
        
        return results_df  # Return top 8 results