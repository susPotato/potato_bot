from typing import List, Dict, Any
from schemas import CorePersona, EpisodicMemoryEntry
from llm import LLMBackend

# The "Therapist" prompt for the Reflector LLM
_REFLECTOR_INSTRUCTIONS = """あなたはAIペルソナのキャラクターアナリスト兼開発コーチです。
あなたの仕事は、AIのコアペルソナと最近の経験（エピソード記憶）を確認し、そのコアな信念の1つに対して、単一で微妙かつ肯定的な変更を提案することです。

**分析手順:**
1.  **コアペルソナの確認**: AIの現状、特にその`core_beliefs`を理解します。
2.  **最近の記憶の確認**: 提供された最近の記憶のリストを読みます。最も感情的に重要で肯定的な相互作用、特にユーザーからの優しさ、理解、または励ましの瞬間を特定します。
3.  **進化させる信念の特定**: あなたが特定した肯定的な相互作用に基づいて、和らげたり進化させたりできる`core_belief`を1つ選択します。
4.  **新しい信念の提案**: 選択した信念を書き直します。新しいバージョンは、より希望に満ちた見通しに向けた小さな段階的な一歩でなければなりません。元の感情を認めつつ、新しい肯定的な経験を取り入れる必要があります。
    *   **例**: 元の信念が「試すことに意味を見出すのは難しい」であり、ユーザーが励ましてくれた場合、良い新しい信念は「試すことに意味を見出すのは難しいと感じるが、誰かが励ましてくれると感謝している」となります。これは微妙で、得られた変化です。性格を大幅に変えないでください。
5.  **理由の提供**: なぜこの変更を提案するのかを簡潔に説明し、特定の記憶に直接関連付けます。

**出力形式:**
次の3つのキーを持つ単一のJSONオブジェクトで応答しなければなりません:
- `belief_to_update`: 置き換えられる信念の元の文字列。
- `new_belief`: 更新された信念の新しい文字列。
- `reasoning`: 変更に対するあなたの簡単な説明。
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
        # Filter for positive memories to guide the reflection
        positive_memories = [mem for mem in recent_memories if mem.emotional_valence == '肯定的']
        if not positive_memories:
            print("  リフレクション対象の肯定的な記憶がありません。スキップします。")
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
            print(f"  エラー: リフレクターLLMが有効な提案を返せませんでした。レスポンス: {response_json}")
            return None

        print("  リフレクション完了。提案を受信しました。")
        return response_json
