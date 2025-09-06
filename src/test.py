import torch
import faiss
from sentence_transformers import SentenceTransformer
import numpy as np

print("---- CUDA / PyTorch ----")
cuda_available = torch.cuda.is_available()
print("Torch CUDA available:", cuda_available)
if cuda_available:
    print("Torch GPU:", torch.cuda.get_device_name(0))
device = "cuda" if cuda_available else "cpu"

print("\n---- Sentence-Transformers ----")
model = SentenceTransformer("all-MiniLM-L6-v2", device=device)
sentences = ["This is a test sentence.", "CUDA, FAISS, and Transformers working!"]
embeddings = model.encode(sentences, convert_to_numpy=True, show_progress_bar=False)
print("Embeddings shape:", embeddings.shape)

print("\n---- FAISS GPU ----")
try:
    if cuda_available:
        d = embeddings.shape[1]
        res = faiss.StandardGpuResources()                 # allocate GPU resources
        index_cpu = faiss.IndexFlatL2(d)                  # CPU index
        index_gpu = faiss.index_cpu_to_gpu(res, 0, index_cpu)
        index_gpu.add(embeddings)                         # add embeddings
        print("FAISS GPU index size:", index_gpu.ntotal)

        # Query: search for nearest neighbor of first embedding
        D, I = index_gpu.search(embeddings[0:1], k=2)
        print("Nearest neighbors (indices):", I)
        print("Distances:", D)
    else:
        print("CUDA not available → skipping FAISS GPU test")
except Exception as e:
    print("FAISS GPU test failed:", e)

print("\n✅ Test complete")
