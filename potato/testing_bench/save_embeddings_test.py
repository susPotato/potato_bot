import torch
from sentence_transformers import SentenceTransformer
from fast_bunkai import FastBunkai
import numpy as np
import pickle
import os

# --- Configuration ---
EMBEDDING_MODEL = "cl-nagoya/ruri-v3-70m"
# Build path relative to the script's location to make it runnable from anywhere
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_FILE = os.path.join(SCRIPT_DIR, "memory", "potato_test_embeddings.pkl")

def main():
    """
    Main function to process a document, generate embeddings, and save them.
    """
    print("--- Save Embeddings Test ---")

    # --- 1. Load Models ---
    print(f"\n[1/4] Loading models...")
    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"      Using device: {device}")
        
        embedding_model = SentenceTransformer(EMBEDDING_MODEL, device=device)
        print(f"      Embedding model '{EMBEDDING_MODEL}' loaded.")
        
        splitter = FastBunkai()
        print(f"      Text splitter 'fast-bunkai' loaded.")

    except Exception as e:
        print(f"\n❌ Error loading models: {e}")
        print("   Please ensure you have the required libraries installed (`pip install -r requirements.txt`).")
        return

    # --- 2. Prepare and Chunk Document ---
    print("\n[2/4] Preparing and chunking document...")
    
    document = """
    ポテトに関する個人的な情報です。
    ポテトは29歳です。彼はソフトウェア開発者として働いています。
    趣味はビデオゲームをプレイすることと、週末にハイキングに行くことです。
    彼の好きな食べ物はラーメンで、特に豚骨ラーメンが好きです。
    彼はいつか日本を旅行して、本場のラーメンを食べることを夢見ています。
    """.strip()
    
    chunks = list(splitter(document))
    print(f"      Document split into {len(chunks)} sentences.")

    # --- 3. Embed Document Chunks ---
    print("\n[3/4] Embedding document chunks...")
    
    # Ruri v3 uses a prefix for semantic search. The empty string "" is used.
    chunk_embeddings = embedding_model.encode(
        ["" + chunk for chunk in chunks],
        convert_to_tensor=False # We want numpy arrays for saving
    )
    print(f"      Created {len(chunk_embeddings)} embeddings.")

    # We will save a list of tuples, where each tuple is (sentence_text, embedding_vector)
    data_to_save = list(zip(chunks, chunk_embeddings))

    # --- 4. Save to File ---
    print(f"\n[4/4] Saving data to '{OUTPUT_FILE}'...")
    try:
        with open(OUTPUT_FILE, 'wb') as f:
            pickle.dump(data_to_save, f)
        
        file_size = os.path.getsize(OUTPUT_FILE)
        print(f"      Successfully saved embeddings.")
        print(f"      File size: {file_size / 1024:.2f} KB")
        print("\nTest complete. You can now inspect the generated pickle file.")

    except Exception as e:
        print(f"\n❌ Error saving file: {e}")


if __name__ == "__main__":
    main()
