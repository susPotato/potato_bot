import json
from typing import List, Dict, Any

from .schemas import Memory, CurationResult
from .models import MODELS
from llm import LLMBackend
from .store import MemoryStore

# The powerful prompt inspired by LangMem's source code
_MEMORY_INSTRUCTIONS = """You are a long-term memory manager maintaining a core store of semantic, procedural, and episodic memory. Your goal is to create a dense, accurate, and useful knowledge base for a life-long learning agent.

Analyze the provided conversation turn and the existing memories that were retrieved as relevant. Reflect on this information and decide what actions to take.

1.  **Extract & Contextualize**:
    *   Identify essential new facts, relationships, preferences, or important details from the conversation.
    *   Do NOT simply copy the conversation. Synthesize the information into a well-written, standalone memory.
    *   For example, if the user says "I love ramen, especially tonkotsu", a good memory is "User's favorite food is ramen, with a preference for tonkotsu."

2.  **Compare & Update**:
    *   Does the new information contradict or update an existing memory? If an old memory is now incorrect (e.g., "User likes dark mode" and they now say "I prefer light mode"), it should be removed.
    *   Does the new information add detail to an existing memory? If so, the old memory should be removed and a new, more detailed one should be created. Do not create overlapping or redundant memories.

3.  **Output Format**:
    *   You MUST respond with a single JSON object.
    *   The JSON object must have two keys: `memories_to_add` and `ids_to_remove`.
    *   `memories_to_add` should be a list of strings, where each string is a new, complete, standalone memory to be saved.
    *   `ids_to_remove` should be a list of strings, where each string is the ID of an existing memory that should be deleted.
"""

class Curator:
    """
    Analyzes conversation history and decides how to update the memory store.
    """
    def __init__(self, llm_backend: LLMBackend):
        self.llm = llm_backend

    def curate_memories(self, conversation_turn: Dict[str, str], existing_memories: List[Memory]) -> CurationResult:
        """
        Uses an LLM to decide which memories to add, update, or remove.
        """
        print("\n---  Curating memories... ---")

        # Prepare the context for the LLM
        prompt_context = f"""
        <conversation_turn>
        User: {conversation_turn['user']}
        Assistant: {conversation_turn['assistant']}
        </conversation_turn>
        """

        if existing_memories:
            prompt_context += "\n<existing_memories>\n"
            for mem in existing_memories:
                prompt_context += f'- ID: {mem.id}, Content: "{mem.content}"\n'
            prompt_context += "</existing_memories>"
        else:
            prompt_context += "\n<existing_memories>\n- None\n</existing_memories>"

        # Call the LLM with the powerful prompt
        response_str = self.llm.call(
            system=_MEMORY_INSTRUCTIONS,
            prompt=prompt_context,
            temperature=0.2 # Low temperature for factual, structured output
        )

        try:
            # Clean the response and parse the JSON.
            # The model often returns conversational text around the JSON block.
            # We find the first '{' and the last '}' to extract the JSON object.
            start = response_str.find('{')
            end = response_str.rfind('}') + 1
            if start != -1 and end != 0:
                json_str = response_str[start:end]
                response_json = json.loads(json_str)
            else:
                raise json.JSONDecodeError("No JSON object found in response.", response_str, 0)


            # Create new Memory objects with embeddings
            memories_to_add = []
            for content in response_json.get("memories_to_add", []):
                embedding = MODELS.embedding_model.encode(content).tolist()
                memories_to_add.append(Memory(content=content, embedding=embedding))
            
            ids_to_remove = response_json.get("ids_to_remove", [])
            
            print(f"   LLM decided to add {len(memories_to_add)} memories and remove {len(ids_to_remove)}.")
            return CurationResult(memories_to_add=memories_to_add, ids_to_remove=ids_to_remove)

        except (json.JSONDecodeError, KeyError) as e:
            print(f" Error parsing curation response from LLM: {e}")
            print(f"   Raw response was: {response_str}")
            return CurationResult()
