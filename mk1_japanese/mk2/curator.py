from typing import List
from schemas import EpisodicMemoryEntry, ConversationTurn
from llm import LLMBackend
from models import MODELS

# A powerful prompt for the Curator LLM
_CURATOR_INSTRUCTIONS = """You are a memory curator for an AI. Your task is to analyze a conversation turn and create a single, insightful, self-contained memory entry.

**Absolute Rules:**
- **Your output MUST be a single JSON object.**
- **You must clearly distinguish between the User's actions and the AI's actions.** Use descriptions like `The user asked...`, `Potato responded...`. Do not conflate the two.

**Analysis Steps:**
1.  **Analyze the provided turn**: Read the user's message and the AI's (Potato's) response. Understand the central topic, emotional tone, and any new information revealed.
2.  **Summarize the memory**: Do not just copy the conversation. Create a concise, third-person summary of the event. Clearly state who took what action.
    *   **Good Example**: `The user asked how Potato was feeling, and Potato responded that it was feeling down.`
    *   **Bad Example**: `The user and Potato shared that they were feeling unwell.`
3.  **Determine Emotional Valence**: Assess the primary emotion of the turn. Label it as one of: `Positive`, `Negative`, or `Neutral`.
4.  **Output Format**: You must respond with a single JSON object with two keys: "curated_memory" (your summarized string) and "emotional_valence" (your chosen label).
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
        print("\n--- 新しい記憶をキュレーション中... ---")

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
            print(f"  エラー: LLMが有効な記憶オブジェクトを返せませんでした。レスポンス: {response_json}")
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
                emotional_valence=response_json.get("emotional_valence", "Neutral"),
                embedding=embedding
            )
            
            print(f"  記憶のキュレーションに成功しました: '{curated_memory_text}'")
            return memory_entry

        except KeyError as e:
            print(f"  エラー: キュレーションのためのLLMレスポンスにキーがありません: {e}")
            return None
        except Exception as e:
            print(f"  記憶作成中に予期せぬエラーが発生しました: {e}")
            return None
