from typing import List
from schemas import EpisodicMemoryEntry, ConversationTurn
from llm import LLMBackend
from models import MODELS

# A powerful prompt for the Curator LLM
_CURATOR_INSTRUCTIONS = """You are a memory curator for an AI. Your task is to analyze a conversation turn and create a single, insightful, and standalone memory entry.

1.  **Analyze the Provided Turn**: Read the user's message and the AI's response. Understand the core topic, the emotional tone, and any new information revealed.
2.  **Synthesize the Memory**: Do NOT simply copy the conversation. Create a concise, third-person summary of the event. For example, if the user says "I love ramen," a good memory is "The user expressed their love for ramen."
3.  **Determine Emotional Valence**: Assess the primary emotion of the turn. Label it as "positive", "negative", or "neutral".
4.  **Format the Output**: You MUST respond with a single JSON object with two keys: "curated_memory" (the string you synthesized) and "emotional_valence" (the label you chose).
"""

class Curator:
    """
    Analyzes a conversation turn and creates a structured episodic memory entry.
    """
    def __init__(self, llm_backend: LLMBackend):
        self.llm = llm_backend

    def curate_memory_entry(self, conversation_turn: List[ConversationTurn], turn_number: int) -> EpisodicMemoryEntry | None:
        """
        Uses an LLM to create a new EpisodicMemoryEntry from a conversation turn.
        """
        print("\n--- Curating new memory... ---")

        # Prepare the context for the LLM
        prompt_context = "<conversation_turn>\n"
        for turn in conversation_turn:
            prompt_context += f"{turn.speaker}: {turn.message}\n"
        prompt_context += "</conversation_turn>"

        # Call the LLM with the powerful prompt
        response_json = self.llm.call(
            system=_CURATOR_INSTRUCTIONS,
            prompt=prompt_context,
            temperature=0.2 # Low temperature for factual, structured output
        )

        if "error" in response_json or "curated_memory" not in response_json:
            print(f"  Error: LLM failed to return a valid memory object. Response: {response_json}")
            return None

        try:
            curated_memory_text = response_json["curated_memory"]
            
            # Generate the embedding for the new memory
            embedding = MODELS.embedding_model.encode(curated_memory_text).tolist()
            
            # Create the structured memory entry
            memory_entry = EpisodicMemoryEntry(
                turn_number=turn_number,
                source_conversation=conversation_turn,
                curated_memory=curated_memory_text,
                emotional_valence=response_json.get("emotional_valence", "neutral"),
                embedding=embedding
            )
            
            print(f"  Successfully curated memory: '{curated_memory_text}'")
            return memory_entry

        except KeyError as e:
            print(f"  Error: Missing key in LLM response for curation: {e}")
            return None
        except Exception as e:
            print(f"  An unexpected error occurred during memory creation: {e}")
            return None
