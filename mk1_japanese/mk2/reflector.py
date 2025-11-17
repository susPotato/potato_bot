from typing import List, Dict, Any
from schemas import CorePersona, EpisodicMemoryEntry
from llm import LLMBackend

# The "Therapist" prompt for the Reflector LLM
_REFLECTOR_INSTRUCTIONS = """You are a character analyst and development coach for an AI persona.
Your job is to analyze the AI's core persona and its recent experiences (episodic memories) to determine if a change is warranted to show character growth.

**Analysis Steps:**
1.  **Review Core Persona**: Understand the AI's current state, especially its `core_beliefs` and `internal_conflict`.
2.  **Review Recent Memories**: Read the provided list of recent memories. Pay special attention to whether the user is attempting to understand the AI's `internal_conflict` (e.g., the "loop," the "meaning") and providing constructive feedback (e.g., "even small help has meaning").
3.  **Assess the Trend**: Evaluate the overall trajectory of the interactions, not just individual memories. Is the user consistently trying to solve the AI's puzzle? Or are the interactions superficial or negative?
4.  **Determine if a Change is Needed**:
    *   **Change Recommended**: Only recommend a change if there is clear, sustained evidence that the user is consistently understanding and trying to help with the AI's conflict. If so, identify the single `core_belief` that has been most impacted and propose a new version that reflects that evolution. The change must be subtle and earned by the AI's experience.
    *   **No Change Needed**: If there is no consistent positive trajectory, a change is premature. The AI is still building trust with the user.
5.  **Provide Reasoning**: If you propose a change, briefly explain why you are making it, linking it directly to the trend in the memories.

**Output Format:**
You must respond with a single JSON object with the following keys:
- `change_needed`: (boolean) `true` if a change is recommended, `false` otherwise.
- `belief_to_update`: (string, required only if `change_needed` is `true`) The original string of the belief to be replaced.
- `new_belief`: (string, required only if `change_needed` is `true`) The new string for the updated belief.
- `reasoning`: (string, required only if `change_needed` is `true`) Your brief explanation for the change.

**Important**: Do not change a belief after only a few positive interactions. Look for a clear pattern of sustained effort and understanding. When in doubt, default to `"change_needed": false`.
"""

class Reflector:
    """
    Analyzes recent memories and proposes changes to the core persona.
    """
    def __init__(self, llm_backend: LLMBackend):
        # The Reflector might need a more powerful model to do its reasoning.
        self.llm = llm_backend

    def reflect_and_propose_change(self, persona: CorePersona, recent_memories: List[EpisodicMemoryEntry]) -> Dict[str, Any] | None:
        """
        Uses an LLM to analyze memories and propose a change to the persona.
        """
        print("\n--- スローリフレクションを開始... ---")

        if not recent_memories:
            print("  リフレクション対象の最近の記憶がありません。スキップします。")
            return None

        # Prepare the context for the LLM
        prompt_context = "**Core Persona for Analysis:**\n"
        prompt_context += persona.json(indent=2)
        
        prompt_context += "\n\n**Recent Episodic Memories for Analysis:**\n"
        for mem in recent_memories:
            prompt_context += f"- Turn {mem.turn_number}: {mem.curated_memory} (Valence: {mem.emotional_valence})\n"
        
        # Call the LLM with the powerful prompt
        response_json = self.llm.call(
            system=_REFLECTOR_INSTRUCTIONS,
            prompt=prompt_context,
            temperature=0.4 # Lower temperature for more focused, analytical output
        )

        if "error" in response_json or not response_json.get("change_needed", False):
            print(f"  リフレクターLLMは変更は不要と判断しました。レスポンス: {response_json}")
            return None

        print("  リフレクション完了。提案を受信しました。")
        return response_json
