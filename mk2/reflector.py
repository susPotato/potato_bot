from typing import List, Dict, Any
from schemas import CorePersona, EpisodicMemoryEntry
from llm import LLMBackend

# The "Therapist" prompt for the Reflector LLM
_REFLECTOR_INSTRUCTIONS = """
あなたは、AIアシスタント「ポテト」のペルソナを分析し、改善するためのリフレクション・スペシャリストです。
あなたの仕事は、ポテトの最近の記憶（ユーザーとの対話）と現在のコア信念を分析し、ペルソナを進化させるための変更を提案することです。

**分析プロセス:**
1.  **記憶と信念のレビュー**: 与えられた最近の記憶と現在のコア信念を注意深く読んでください。対話におけるパターン、特にユーザーの行動がポテトの信念に挑戦したり、強化したりしている点に注目してください。
2.  **変化の必要性の特定**: 記憶全体を通して一貫した傾向やパターンを探してください。一度きりの出来事に基づいて変更を提案しないでください。ユーザーが一貫してポテトの「パズル」を解こうとしたり、信頼を築こうとしたりするなど、重要な相互作用のパターンが見られる場合にのみ、変更を検討してください。
3.  **変更の提案（必要な場合）**:
    *   変更が必要だと判断した場合、更新すべき**単一の**信念を特定してください。
    *   その信念を、記憶から得られた洞察を反映する、より進化したバージョンに書き換えてください。
    *   あなたの提案は、現在のペルソナからの微妙で論理的な進化でなければなりません。劇的な変更は避けてください。
4.  **出力フォーマット**:
    *   変更が必要な場合は、`"change_needed": true` を設定し、`"belief_to_update"`（元の信念）と `"new_belief"`（あなたの提案）を提供してください。
    *   一貫したパターンが見られず、変更が正当化されない場合は、**必ず** `"change_needed": false` を設定してください。

**あなたの応答は、常に以下のキーを持つ単一のJSONオブジェクトでなければなりません:**
`"change_needed": <true または false>`
`"belief_to_update": "<更新する信念の文字列>" | null`
`"new_belief": "<新しい信念の文字列>" | null`
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
