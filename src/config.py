import torch

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
MODEL = 'all-mpnet-base-v2' # 'paraphrase-multilingual-mpnet-base-v2'
BATCH_SIZE = 128
PATH_TO_DATA = 'data/movie_details.csv'
INDEX_FILE = 'data/faiss_index.faiss'
INDEX_METRIC = 'IndexFlatIP'  # Options: 'IndexFlatL2', 'IndexFlatIP', 'IndexIVFFlat'.
IVFF_NLIST = 100  # Number of clusters for IndexIVFFlat (if used).
PKL_PATH = 'data/embeddings.pkl'
CROSS_ENCODER = "cross-encoder/ms-marco-electra-base"