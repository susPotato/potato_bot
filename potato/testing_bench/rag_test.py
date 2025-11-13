import torch
import numpy as np
import ollama
from sentence_transformers import SentenceTransformer
from fast_bunkai import FastBunkai

# --- Configuration ---
EMBEDDING_MODEL = "cl-nagoya/ruri-v3-70m"
LLM_MODEL = "llama3:8b" # Make sure this model is pulled in Ollama

def cosine_similarity(v1, v2):
    """Compute cosine similarity between two vectors."""
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

def main():
    """
    Main function to run the RAG test.
    """
    print("--- Japanese RAG and SLM Test ---")

    # --- 1. Load Models ---
    print(f"\n[1/5] Loading models...")
    try:
        # Check if Ollama is running and the model is available
        ollama.list()
        print(f"      Ollama is running.")
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"      Using device: {device}")
        
        embedding_model = SentenceTransformer(EMBEDDING_MODEL, device=device)
        print(f"      Embedding model '{EMBEDDING_MODEL}' loaded.")
        
        splitter = FastBunkai()
        print(f"      Text splitter 'fast-bunkai' loaded.")

    except Exception as e:
        print(f"\n❌ Error loading models: {e}")
        print("   Please ensure Ollama is running (`ollama serve`) and you have the required libraries installed (`pip install -r requirements.txt`).")
        return

    # --- 2. Prepare and Chunk Document ---
    print("\n[2/5] Preparing and chunking document...")
    
    document = """
    日本の首都は東京です。東京は、日本の本州東部に位置する都市で、世界で最も人口の多い大都市圏の一つです。
    多くの文化的な名所があり、浅草の浅草寺、渋谷のスクランブル交差点、そして美味しい寿司で知られています。
    また、日本のアニメや漫画の中心地でもあります。
    """.strip()
    
    chunks = list(splitter(document))
    print(f"      Document split into {len(chunks)} sentences:")
    for i, chunk in enumerate(chunks):
        print(f"        {i+1}: {chunk}")

    # --- 3. Embed Document Chunks ---
    print("\n[3/5] Embedding document chunks...")
    
    # Ruri v3 uses a prefix for semantic search. The empty string "" is used.
    # See: https://huggingface.co/cl-nagoya/ruri-v3-70m
    chunk_embeddings = embedding_model.encode(
        ["" + chunk for chunk in chunks],
        convert_to_tensor=False # We want numpy arrays for CPU-based similarity search
    )
    print(f"      Created {len(chunk_embeddings)} embeddings of dimension {chunk_embeddings.shape[1]}.")

    # --- 4. Perform Vector Search ---
    print("\n[4/5] Performing vector search...")
    query = "アニメについて教えて"
    print(f"      Query: '{query}'")
    
    # For retrieval tasks, Ruri v3 recommends the "検索クエリ: " prefix.
    query_embedding = embedding_model.encode("検索クエリ: " + query)
    
    # Calculate similarities
    similarities = [cosine_similarity(query_embedding, chunk_emb) for chunk_emb in chunk_embeddings]
    
    # Get the top 2 most relevant chunks
    top_k = 2
    top_indices = np.argsort(similarities)[-top_k:][::-1]
    
    retrieved_context = "\n".join([chunks[i] for i in top_indices])
    
    print(f"      Retrieved top {top_k} most relevant sentences as context:")
    print("      --- Context Start ---")
    print(retrieved_context)
    print("      --- Context End ---")

    # --- 5. Generate Response with LLM ---
    print("\n[5/5] Generating response with local LLM...")
    
    prompt = f"""
    以下の文脈に基づいて、ユーザーの質問に日本語で簡潔に答えてください。

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
                {"role": "system", "content": "You are a helpful assistant that answers questions in Japanese based on the provided context."},
                {"role": "user", "content": prompt}
            ]
        )
        
        final_answer = response['message']['content']
        
        print("\n--- FINAL RESPONSE ---")
        print(final_answer)
        print("----------------------")

    except Exception as e:
        print(f"\n❌ Error during LLM generation: {e}")
        print(f"   Please ensure the model '{LLM_MODEL}' is available in Ollama (`ollama pull {LLM_MODEL}`).")

if __name__ == "__main__":
    main()



