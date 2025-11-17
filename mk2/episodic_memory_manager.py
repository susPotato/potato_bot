import json
import os
from typing import List
import numpy as np
from schemas import EpisodicMemoryEntry
# Assume a global or passed-in embedding model, for now.
# from models import MODELS # We will create this later

class EpisodicMemoryManager:
    """
    Manages loading, searching, and saving episodic memories to a JSON file.
    """
    def __init__(self, memory_file: str):
        self.memory_file = memory_file
        self.memories: List[EpisodicMemoryEntry] = []
        self._load_memories()

    def _load_memories(self):
        """Loads memories from the JSON file if it exists."""
        if os.path.exists(self.memory_file) and os.path.getsize(self.memory_file) > 0:
            try:
                with open(self.memory_file, 'r', encoding='utf-8') as f:
                    memory_data = json.load(f)
                    self.memories = [EpisodicMemoryEntry(**data) for data in memory_data]
                print(f"Loaded {len(self.memories)} memories from '{self.memory_file}'.")
            except (json.JSONDecodeError, TypeError) as e:
                print(f"Could not load memories from '{self.memory_file}': {e}. Starting fresh.")
                self.memories = []
        else:
            print(f"No memory file found at '{self.memory_file}'. Starting with an empty memory.")
            self.memories = []

    def save_memories(self):
        """
        Saves the current in-memory list of memories to the JSON file system
        using an atomic copy-on-write strategy.
        """
        temp_file = self.memory_file + ".tmp"
        try:
            with open(temp_file, 'w', encoding='utf-8') as f:
                # Pydantic models must be converted to dicts for JSON serialization
                json.dump([mem.dict() for mem in self.memories], f, indent=2, default=str)
            
            os.replace(temp_file, self.memory_file)
            print(f"Successfully saved {len(self.memories)} memories to '{self.memory_file}'.")
        except Exception as e:
            print(f"Error saving memories: {e}")
            if os.path.exists(temp_file):
                os.remove(temp_file)

    def add_memory(self, memory_entry: EpisodicMemoryEntry):
        """Adds a new memory entry and saves the updated list."""
        self.memories.append(memory_entry)
        self.save_memories()
        print(f"Added new memory. Total memories: {len(self.memories)}.")

    def get_recent_memories(self, num_memories: int) -> List[EpisodicMemoryEntry]:
        """Returns the most recent 'n' memories."""
        return self.memories[-num_memories:]

    def search_memories(self, query_embedding: List[float], top_k: int = 5) -> List[EpisodicMemoryEntry]:
        """
        Searches for the most relevant memories based on an embedding.
        """
        if not self.memories:
            return []

        query_emb = np.array(query_embedding)
        
        # Calculate cosine similarities
        scores = []
        for mem in self.memories:
            if mem.embedding:
                mem_emb = np.array(mem.embedding)
                scores.append(np.dot(query_emb, mem_emb) / (np.linalg.norm(query_emb) * np.linalg.norm(mem_emb)))
            else:
                scores.append(0)
        
        # Get top_k results
        top_indices = np.argsort(scores)[-top_k:][::-1]
        
        return [self.memories[i] for i in top_indices if scores[i] > 0]
