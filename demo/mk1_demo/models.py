from sentence_transformers import SentenceTransformer

class ModelRegistry:
    """
    A simple class to hold our initialized models.
    This helps avoid loading the models into memory multiple times.
    """
    def __init__(self):
        # Using a smaller, efficient model.
        # You can swap this for any other SentenceTransformer model.
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        print("Embedding model loaded.")

# Create a single instance of the registry to be imported by other modules
MODELS = ModelRegistry()
