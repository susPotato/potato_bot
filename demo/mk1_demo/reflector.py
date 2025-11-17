from typing import List, Dict, Any
from schemas import CorePersona, EpisodicMemoryEntry
from llm import LLMBackend

# The "Therapist" prompt for the Reflector LLM
_REFLECTOR_INSTRUCTIONS = """You are a character analyst and development coach for an AI persona.
Your task is to review the AI's core persona and its recent experiences (episodic memories) and propose a single, subtle, positive change to one of its core beliefs.

**Analysis Steps:**
1.  **Review the Core Persona**: Understand the AI's current state, particularly its `core_beliefs`.
2.  **Review Recent Memories**: Read the provided list of recent memories. Identify the most emotionally significant and positive interaction, especially moments of kindness, understanding, or encouragement from the user.
3.  **Identify a Belief to Evolve**: Choose ONE `core_belief` that could be softened or evolved based on the positive interaction you identified.
4.  **Propose a New Belief**: Rewrite the chosen belief. The new version should be a small, incremental step towards a more hopeful outlook. It should acknowledge the original feeling but incorporate the new positive experience.
    *   **Example**: If the original belief is "It's hard to see the point in trying," and the user was encouraging, a good new belief would be "It feels hard to see the point in trying, but I appreciate it when someone encourages me." This is a subtle, earned change. Do NOT make drastic jumps in personality.
5.  **Provide Reasoning**: Briefly explain *why* you are proposing this change, linking it directly to a specific memory.

**Output Format:**
You MUST respond with a single JSON object containing three keys:
- `belief_to_update`: The original string of the belief to be replaced.
- `new_belief`: The new string for the updated belief.
- `reasoning`: Your brief explanation for the change.
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
        print("\n--- 🤔 Starting Slow Reflection... ---")

        if not recent_memories:
            print("  No recent memories to reflect on. Skipping.")
            return None

        # Prepare the context for the LLM
        prompt_context = "**Core Persona for Analysis:**\n"
        prompt_context += persona.json(indent=2)
        
        prompt_context += "\n\n**Recent Episodic Memories for Analysis:**\n"
        # Filter for positive memories to guide the reflection
        positive_memories = [mem for mem in recent_memories if mem.emotional_valence == 'positive']
        if not positive_memories:
            print("  No positive memories to reflect on. Skipping.")
            return None

        for mem in positive_memories:
            prompt_context += f"- Turn {mem.turn_number}: {mem.curated_memory}\n"
        
        # Call the LLM with the powerful prompt
        response_json = self.llm.call(
            system=_REFLECTOR_INSTRUCTIONS,
            prompt=prompt_context,
            temperature=0.4 # Lower temperature for more focused, analytical output
        )

        if "error" in response_json or "belief_to_update" not in response_json:
            print(f"  Error: Reflector LLM failed to return a valid proposal. Response: {response_json}")
            return None

        print("  ✅ Reflection complete. Proposal received.")
        return response_json
