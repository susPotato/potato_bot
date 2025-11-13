import os
import shutil
import sys

# --- Path Setup ---
# This ensures that the script can find the rag_system and llm modules
# by adding the parent 'Potato' directory to the Python path.
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

from rag_system.store import MemoryStore
from rag_system.curator import Curator
from llm import LLMBackend

# --- Configuration ---
MEMORY_DIR = os.path.join(SCRIPT_DIR, "rag_system_memory")
MEMORY_FILE = os.path.join(MEMORY_DIR, "test_memories.pkl")
LLM_MODEL = "llama3:8b"

def main():
    """
    An end-to-end test for the new RAG system.
    Simulates a conversation, curates memories after each turn,
    and saves the final result.
    """
    print("---  RAG System End-to-End Test ---")

    # --- 1. Setup ---
    # Clean up previous test runs
    if os.path.exists(MEMORY_DIR):
        shutil.rmtree(MEMORY_DIR)
    os.makedirs(MEMORY_DIR)
    
    print(f"Using memory file: {MEMORY_FILE}")

    try:
        llm_backend = LLMBackend(model_name=LLM_MODEL)
        # Verify LLM connection
        llm_backend.call("hi", "hi") 
    except Exception as e:
        print(f"❌ Could not connect to LLM backend: {e}")
        print("   Please ensure Ollama is running and the model is pulled.")
        return

    store = MemoryStore(memory_file=MEMORY_FILE)
    curator = Curator(llm_backend=llm_backend)

    # --- 2. Simulated Conversation (in English) ---
    conversation_history = [
        {
            "user": "Hello! My name is Potato.",
            "assistant": "Hello Potato! It's nice to meet you."
        },
        {
            "user": "I love Japanese ramen, especially the tonkotsu style.",
            "assistant": "That's great! Tonkotsu ramen is delicious."
        },
        {
            "user": "By the way, I prefer to use light mode for my UI.",
            "assistant": "Got it. I'll remember that you prefer light mode."
        },
        {
            "user": "Actually, I've changed my mind. Dark mode is easier on my eyes.",
            "assistant": "No problem. I'll update your preference to dark mode."
        }
    ]

    # --- 3. Curation Loop ---
    for i, turn in enumerate(conversation_history):
        print(f"--- Processing Turn {i+1}/{len(conversation_history)} ---")
        
        # a. Search for relevant memories
        # We search based on a summary of the current turn
        query = f"User: {turn['user']}\\nAssistant: {turn['assistant']}"
        relevant_memories = store.search_memories(query, top_k=5)
        print(f"Found {len(relevant_memories)} relevant memories for the current turn.")

        # b. Curate memories based on the new turn and relevant context
        curation_result = curator.curate_memories(turn, relevant_memories)

        # c. Apply the curator's decisions to the store
        store.apply_updates(
            memories_to_add=curation_result.memories_to_add,
            ids_to_remove=curation_result.ids_to_remove
        )

    # --- 4. Final Save ---
    print("\n--- ✅ Conversation Finished ---")
    store.save_memories()

    # --- 5. Verification ---
    print("\n--- Verifying final memory store ---")
    final_store = MemoryStore(memory_file=MEMORY_FILE)
    
    # We expect the final memory to contain the updated preference for dark mode
    # and not the old light mode preference.
    search_results = final_store.search_memories("What is the user's UI preference?", top_k=3)
    
    print("\nSearch results for 'What is the user's UI preference?':")
    if not search_results:
        print("   No relevant memories found.")
    for mem in search_results:
        print(f"   - ID: {mem.id}, Content: '{mem.content}'")

if __name__ == "__main__":
    main()
