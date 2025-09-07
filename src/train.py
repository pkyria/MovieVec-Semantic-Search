import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import pickle
import torch
import os.path


# Import configurations (MODEL, BATCH_SIZE, DEVICE)
from config import *

class Embedder:
    def __init__(self, texts, model_name=MODEL):
        self.model = SentenceTransformer(model_name)
        self.dimension = None # Initialize dimension to None
        self.ids = self.load_ids() # Load existing IDs if available

    def load_ids(self):
        """ Load existing IDs from pickle file. """
        try:
            with open(PKL_PATH, "rb") as f:
                data = pickle.load(f)
                ids= data["ids"]
                print(f"Loading existing IDs with count: {len(ids)}")
                return ids
            
        except (FileNotFoundError, KeyError):
            print(f"No existing IDs found.")
            return None    

        
    def encode(self, texts):
        """ Generate (or store existing) embeddings for a list of texts. """
        try:
            with open(PKL_PATH, "rb") as f:
                data = pickle.load(f)
                embeddings = data["embeddings"]
                ids= data["ids"]
                print(f"Loading existing embeddings with shape: {embeddings.shape[0]},{embeddings.shape[1]}")
                self.dimension = embeddings.shape[1] # Set the dimension
                return embeddings, ids
            
        except (FileNotFoundError, KeyError):
            print(f"Generating embeddings for {len(texts)} texts...")
            os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True' # Allow PyTorch to expand memory segments as needed

            chunk_size = 200000 # Process texts in chunks to manage memory usage
            all_embeddings = []
            for i in range(0, len(texts), chunk_size):
                chunk = texts[i:i+chunk_size]
                print(f"Processing chunk {i//chunk_size + 1} with {len(chunk)} texts...")

                # Clear GPU cache before each chunk
                if DEVICE == 'cuda':
                    torch.cuda.empty_cache()
                    
                    # Check available memory
                    free_memory = torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_reserved()
                    print(f"Free GPU memory: {free_memory / 1024**3:.2f} GB")
                
                
                # Encode chunk with conservative settings
                chunk_embeddings = self.model.encode(
                    chunk,
                    batch_size=BATCH_SIZE,
                    show_progress_bar=True,
                    device=DEVICE,
                    convert_to_numpy=True  # Convert to numpy immediately to save GPU memory (moves to CPU)
                )
                
                # Ensure embeddings are numpy arrays
                if isinstance(chunk_embeddings, torch.Tensor): # If embeddings are torch tensors, move to CPU and convert to numpy
                    chunk_embeddings = chunk_embeddings.cpu().numpy()
                
                all_embeddings.append(chunk_embeddings)
                
                print(f"Chunk processed successfully. Shape: {chunk_embeddings.shape}")
                
                # Clear GPU memory after each chunk
                if DEVICE == 'cuda':
                    torch.cuda.empty_cache()
                    
                    # Show memory usage
                    peak_memory = torch.cuda.max_memory_allocated() / 1024**3
                    current_memory = torch.cuda.memory_allocated() / 1024**3
                    print(f"Peak GPU memory: {peak_memory:.2f} GB")
                    print(f"Current GPU memory: {current_memory:.2f} GB")
                    torch.cuda.reset_peak_memory_stats()

            print("All chunks processed.")
            embeddings = np.vstack(all_embeddings) # Combine all chunk embeddings into a single array
            ids = np.arange(len(texts)) # Generate IDs for texts

            # texts = texts[:39000] # Limit to first 10000 texts for testing
            
            # embeddings = self.model.encode(
            #     texts, 
            #     convert_to_tensor=True,
            #     batch_size= BATCH_SIZE,
            #     show_progress_bar= True,
            #     device= DEVICE)

            print(f"Generated embeddings with shape: {embeddings.shape}")
            self.dimension = embeddings.shape[1] # Set the dimension
            # save everything to pickle
            with open(PKL_PATH, "wb") as f:
                pickle.dump({"embeddings": embeddings, "ids": ids}, f)
            if isinstance(embeddings, torch.Tensor):   
                return embeddings.cpu().numpy(), ids # Move embeddings to CPU (if on GPU) and convert to numpy array for FAISS
            else:
                return embeddings, ids
        
        

class FaissIndex:
    def __init__(self, **kwargs):
        """ Initialize the FAISS index. """
        dimension = kwargs.get('dimension', None)
        print(f"Initializing FAISS index with dimension: {dimension}")
        if INDEX_METRIC == 'IndexFlatL2':
            self.index_cpu = faiss.IndexFlatL2(dimension)  # Using L2 distance metric
        elif INDEX_METRIC == 'IndexFlatIP':
            self.index_cpu = faiss.IndexFlatIP(dimension)  # Using Inner Product metric
        elif INDEX_METRIC == 'IndexIVFFlat':
            quantizer = faiss.IndexFlatIP(dimension)
            self.index_cpu = faiss.IndexIVFFlat(quantizer, dimension, IVFF_NLIST)

    def add_embeddings(self, embeddings):
        """ Add embeddings to the FAISS index. """
        if len( embeddings ) > 0:
            print(f"Adding {embeddings.shape[0]} embeddings of dimension {embeddings.shape[1]} to the index.")
        else:
            print("No embeddings to add to the index.")
            return
        faiss.normalize_L2(embeddings)  # Normalize embeddings for cosine similarity
        # Train the index if necessary (for IndexIVFFlat)
        if not self.index_cpu.is_trained and INDEX_METRIC == 'IndexIVFFlat':
            self.index_cpu.train(embeddings)
        

        self.index_cpu.add(embeddings.astype('float32'))  # Ensure embeddings are float32
        print(f"Total embeddings in index: {self.index_cpu.ntotal}")
         

    def save_index(self, file_path):
        """ Save the FAISS index to a file. """
        print(f"Saving index to file...")
        faiss.write_index(self.index_cpu, INDEX_FILE)
        print(f"Index saved to {INDEX_FILE}")

    def load_index(self, file_path):
        """ Load a FAISS index from a file. """
        print(f"Loading index from file...")
        
        # Check if index file exists before loading it 
        if not os.path.exists(INDEX_FILE):
            print(f"Index file {INDEX_FILE} does not exist.")
            return
        else: # Load the index from file 
            self.index_cpu = faiss.read_index(INDEX_FILE, faiss.IO_FLAG_MMAP) # Memory-map the index file
            print("Index size (ntotal):", self.index_cpu.ntotal)
            self.index = self.index_cpu

            # # Move to GPU if available (uncomment if fixed)
            # if DEVICE == 'cuda':
            #     self.index = self.move_to_gpu(self.index_cpu)
            # else:
            #     self.index = self.index_cpu

        if self.index:
            print(f"Index loaded from {INDEX_FILE}")

            return self.index
        else:
            print("Failed to load index.")
            return
        
    def move_to_gpu(self, index_cpu):
        """ Move the FAISS index to GPU (if available). """
        res = faiss.StandardGpuResources()                 # allocate GPU resources
        index_gpu = faiss.index_cpu_to_gpu(res, 0, index_cpu)
        print("FAISS index moved to GPU.")
        return index_gpu
