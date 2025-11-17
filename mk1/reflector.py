from typing import List, Dict, Any
from schemas import CorePersona, EpisodicMemoryEntry
from llm import LLMBackend

# The "Therapist" prompt for the Reflector LLM
_REFLECTOR_INSTRUCTIONS = """あなたはAIペルソナのキャラクターアナリスト兼開発コーチです。
あなたの仕事は、AIのコアペルソナと最近の経験（エピソード記憶）を分析し、ペルソナの成長を示すための変更が必要かどうかを判断することです。

**分析手順:**
1.  **コアペルソナの確認**: AIの現状、特にその`core_beliefs`と`internal_conflict`を理解します。
2.  **最近の記憶の確認**: 提供された最近の記憶のリストを読みます。ユーザーがAIの`internal_conflict`（例：「ループ」「意味」）を理解しようとし、それに対して建設的なフィードバック（例：「小さな助けでも意味がある」）を提供しているかどうかに特に注意してください。
3.  **トレンドの評価**: 個々の記憶だけでなく、相互作用の全体的な軌跡を評価します。ユーザーは一貫してAIのパズルを解こうとしていますか？それとも、相互作用は表面的または否定的ですか？
4.  **変更の判断**:
    *   **変更が必要な場合**: ユーザーが一貫してAIの葛藤を理解し、助けようとしているという明確で持続的な証拠がある場合にのみ、変更を推奨します。その場合、最も影響を受けた`core_belief`を1つ特定し、その進化を反映する新しいバージョンを提案します。変更は微妙で、AIの経験に基づいている必要があります。
    *   **変更が不要な場合**: 一貫した肯定的な軌跡がない場合、変更は時期尚早です。AIはまだユーザーとの信頼を築いている途中です。
5.  **理由の提供**: 変更を提案する場合、なぜこの変更を行うのかを簡潔に説明し、特定の記憶の傾向に直接関連付けます。

**出力形式:**
次のキーを持つ単一のJSONオブジェクトで応答しなければなりません:
- `change_needed`: (boolean) 変更が推奨される場合は`true`、そうでない場合は`false`。
- `belief_to_update`: (string, `change_needed`が`true`の場合にのみ必須) 置き換えられる信念の元の文字列。
- `new_belief`: (string, `change_needed`が`true`の場合にのみ必須) 更新された信念の新しい文字列。
- `reasoning`: (string, `change_needed`が`true`の場合にのみ必須) 変更に対するあなたの簡単な説明。

**重要**: ユーザーからの数回の肯定的なやり取りだけで信念を変更しないでください。持続的な努力と理解の明確なパターンを探してください。疑わしい場合は、`"change_needed": false`と設定してください。
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
        prompt_context = "**分析対象のコアペルソナ:**\n"
        prompt_context += persona.json(indent=2)
        
        prompt_context += "\n\n**分析対象の最近のエピソード記憶:**\n"
        for mem in recent_memories:
            prompt_context += f"- Turn {mem.turn_number}: {mem.curated_memory} (感情価: {mem.emotional_valence})\n"
        
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
