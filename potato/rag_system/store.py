import os
import pickle
import numpy as np
from typing import List, Tuple

from .schemas import Memory
from .models import MODELS

def cosine_similarity(v1, v2):
    """Compute cosine similarity between two vectors."""
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

class MemoryStore:
    """
    Manages the loading, searching, and saving of memories to a persistent file.
    Implements atomic writes to prevent data corruption.
    """
    def __init__(self, memory_file: str):
        self.memory_file = memory_file
        self.memories: List[Memory] = []
        self._load_memories()

    def _load_memories(self):
        """Loads memories from the pickle file if it exists."""
        if os.path.exists(self.memory_file):
            try:
                with open(self.memory_file, 'rb') as f:
                    self.memories = pickle.load(f)
                print(f" Loaded {len(self.memories)} memories from '{self.memory_file}'.")
            except (pickle.UnpicklingError, EOFError) as e:
                print(f" Could not load memories from '{self.memory_file}': {e}. Starting fresh.")
                self.memories = []
        else:
            print(f"No memory file found at '{self.memory_file}'. Starting fresh.")
            self.memories = []

    def search_memories(self, query: str, top_k: int = 5) -> List[Memory]:
        """
        Searches for the most relevant memories based on a text query.
        """
        if not self.memories:
            return []

        query_embedding = MODELS.embedding_model.encode(query, convert_to_numpy=True)
        
        # Calculate similarities
        scores = []
        for mem in self.memories:
            if mem.embedding:
                scores.append(cosine_similarity(query_embedding, np.array(mem.embedding)))
            else:
                scores.append(0)
        
        # Get top_k results
        top_indices = np.argsort(scores)[-top_k:][::-1]
        
        return [self.memories[i] for i in top_indices if scores[i] > 0]

    def apply_updates(self, memories_to_add: List[Memory], ids_to_remove: List[str]):
        """
        Applies updates to the in-memory list of memories.
        """
        # Remove memories
        initial_count = len(self.memories)
        self.memories = [mem for mem in self.memories if mem.id not in ids_to_remove]
        removed_count = initial_count - len(self.memories)

        # Add new memories
        self.memories.extend(memories_to_add)
        added_count = len(memories_to_add)

        print(f"Memory updates applied: {added_count} added, {removed_count} removed.")

    def save_memories(self):
        """
        Saves the current in-memory list of memories to the file system
        using an atomic copy-on-write strategy.
        """
        temp_file = self.memory_file + ".tmp"
        try:
            # Write to a temporary file first
            with open(temp_file, 'wb') as f:
                pickle.dump(self.memories, f)
            
            # Atomically rename the temp file to the final file
            os.replace(temp_file, self.memory_file)
            print(f" Successfully saved {len(self.memories)} memories to '{self.memory_file}'.")

        except Exception as e:
            print(f" Error saving memories: {e}")
            if os.path.exists(temp_file):
                os.remove(temp_file)
