import torch
from sentence_transformers import SentenceTransformer
from fast_bunkai import FastBunkai

# --- Configuration ---
# Using a standard, high-performance English embedding model for this test
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

class Models:
    """
    A singleton class to load and hold the language models, ensuring they
    are only loaded into memory once.
    """
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            print("--- Loading language models... ---")
            cls._instance = super(Models, cls).__new__(cls)
            
            cls.device = "cuda" if torch.cuda.is_available() else "cpu"
            print(f"      Using device: {cls.device}")
            
            try:
                cls.embedding_model = SentenceTransformer(EMBEDDING_MODEL, device=cls.device)
                print(f"      Embedding model '{EMBEDDING_MODEL}' loaded.")
                
                cls.splitter = FastBunkai()
                print("      Text splitter 'fast-bunkai' loaded.")
            except Exception as e:
                print(f" Error loading models: {e}")
                print("   Please ensure you have run 'pip install -r requirements.txt'")
                cls._instance = None
        return cls._instance

# To be imported by other modules
MODELS = Models()
