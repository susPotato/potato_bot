import torch
from sentence_transformers import SentenceTransformer
from fast_bunkai import FastBunkai
import numpy as np
import pickle
import os

# --- Configuration ---
EMBEDDING_MODEL = "cl-nagoya/ruri-v3-70m"
OUTPUT_FILE = os.path.join("memory", "potato_test_embeddings.pkl")

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
    日本の首都は東京です。東京は、日本の本州東部に位置する都市で、世界で最も人口の多い大都市圏の一つです。
    多くの文化的な名所があり、浅草の浅草寺、渋谷のスクramble交差点、そして美味しい寿司で知られています。
    また、日本のアニメや漫画の中心地でもあります。
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
