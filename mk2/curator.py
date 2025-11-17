from typing import List
from schemas import EpisodicMemoryEntry, ConversationTurn
from llm import LLMBackend
from models import MODELS

# A powerful prompt for the Curator LLM
_CURATOR_INSTRUCTIONS = """
あなたは会話分析の専門家です。ユーザーとAIアシスタント「ポテト」との会話のターンが与えられます。
あなたの仕事は、このターンを簡潔で三人称の、過去形の事実に基づいた記憶として要約することです。

重要なルール:
- 常に三人称の視点を使用してください（例：「ユーザーは尋ねた」「ポテトは答えた」）。
- ユーザーとポテトの行動を明確に区別してください。
- 感情的な評価は「neutral」に設定してください。

例:
- ターン: [ユーザー: "調子はどう？", ポテト: "あまり良くないです。"]
- あなたの出力: { "curated_memory": "ユーザーはポテトの調子を尋ね、ポテトは気分が良くないと答えた。", "emotional_valence": "neutral" }

- ターン: [ユーザー: "あなたについてもっと教えて。", ポテト: "私はポテトです。"]
- あなたの出力: { "curated_memory": "ユーザーはポテトについてもっと知りたいと頼み、ポテトは自己紹介をした。", "emotional_valence": "neutral" }

あなたの応答は、以下のキーを持つ単一のJSONオブジェクトでなければなりません:
"curated_memory": "要約された記憶"
"emotional_valence": "neutral"
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
