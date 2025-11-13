import torch
import numpy as np
import ollama
from sentence_transformers import SentenceTransformer
import pickle
import os

# --- Configuration ---
EMBEDDING_MODEL = "cl-nagoya/ruri-v3-70m"
LLM_MODEL = "llama3:8b" # Make sure this model is pulled in Ollama

# --- Build path relative to the script's location ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
EMBEDDINGS_FILE = os.path.join(SCRIPT_DIR, "memory", "potato_test_embeddings.pkl")

def cosine_similarity(v1, v2):
    """Compute cosine similarity between two vectors."""
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

def main():
    """
    Main function to load embeddings, perform RAG, and extract information.
    """
    print("--- Information Extraction RAG Test ---")

    # --- 1. Load Models and Saved Embeddings ---
    print(f"\n[1/4] Loading models and embeddings...")
    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"      Using device: {device}")
        
        embedding_model = SentenceTransformer(EMBEDDING_MODEL, device=device)
        print(f"      Embedding model '{EMBEDDING_MODEL}' loaded.")
        
        with open(EMBEDDINGS_FILE, 'rb') as f:
            saved_data = pickle.load(f)
        
        # Unzip the saved data into separate lists for texts and embeddings
        db_texts, db_embeddings = zip(*saved_data)
        print(f"      Loaded {len(db_texts)} embeddings from '{EMBEDDINGS_FILE}'.")

    except FileNotFoundError:
        print(f"\n❌ Error: Embeddings file not found at '{EMBEDDINGS_FILE}'.")
        print("   Please run 'save_embeddings_test.py' first to generate the file.")
        return
    except Exception as e:
        print(f"\n❌ Error loading models or file: {e}")
        return

    # --- 2. Perform Vector Search ---
    print("\n[2/4] Performing vector search...")
    query = "ポテトは何歳ですか？" # "How old is Potato?"
    print(f"      Query: '{query}'")
    
    # Embed the query
    query_embedding = embedding_model.encode("検索クエリ: " + query)
    
    # Calculate similarities
    similarities = [cosine_similarity(query_embedding, emb) for emb in db_embeddings]
    
    # Get the top 1 most relevant chunk
    top_k = 1
    top_index = np.argmax(similarities)
    
    retrieved_context = db_texts[top_index]
    
    print(f"      Retrieved the most relevant sentence with a similarity of {similarities[top_index]:.4f}:")
    print("      --- Context Start ---")
    print(f"      '{retrieved_context}'")
    print("      --- Context End ---")

    # --- 3. Generate Response with LLM ---
    print("\n[3/4] Generating response with local LLM...")
    
    prompt = f"""
    以下の文脈のみに基づいて、ユーザーの質問に日本語で簡潔に答えてください。

    文脈:
    {retrieved_context}

    質問:
    {query}
    
    回答:
    """

    print(f"      Sending prompt to '{LLM_MODEL}'...")
    try:
        response = ollama.chat(
            model=LLM_MODEL,
            messages=[
                {"role": "user", "content": prompt}
            ]
        )
        final_answer = response['message']['content']
        
        print("\n[4/4] Final Answer from LLM:")
        print("      " + final_answer)

    except Exception as e:
        print(f"\n❌ Error during LLM generation: {e}")
        print(f"   Please ensure Ollama is running and the model '{LLM_MODEL}' is available.")

if __name__ == "__main__":
    main()
