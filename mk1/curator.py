from typing import List
from schemas import EpisodicMemoryEntry, ConversationTurn
from llm import LLMBackend
from models import MODELS

# A powerful prompt for the Curator LLM
_CURATOR_INSTRUCTIONS = """あなたはAIのための記憶キュレーターです。あなたのタスクは、会話のターンを分析し、洞察に満ちた、自己完結した単一の記憶エントリを作成することです。

**絶対に守るべきルール:**
- **すべての出力、特に `curated_memory` は、必ず日本語でなければなりません。**
- **応答は単一のJSONオブジェクトでなければなりません。**
- **ユーザーとAIの行動を明確に区別すること。** `ユーザーは尋ねた`、`ポテトは答えた`のように、誰が何をしたかを明確に記述してください。両者を混同しないでください。

**分析手順:**
1.  **提供されたターンの分析**: ユーザーのメッセージとAI（ポテト）の応答を読んでください。中心的なトピック、感情的なトーン、そして明らかにされた新しい情報を理解してください。
2.  **記憶の要約（日本語で）**: 会話を単にコピーしないでください。イベントを簡潔な第三者の要約として、**必ず日本語で**作成してください。誰が行動を起こしたかを明確に示してください。
    *   **良い例**: `ユーザーはポテトの調子を尋ねたが、ポテトは気分が落ち込んでいると答えた。`
    *   **悪い例**: `ユーザーとポテトは、調子が悪く感じていることを共有した。`
3.  **感情価の決定**: ターンの主要な感情を評価してください。「肯定的」、「否定的」、「中立」のいずれかでラベル付けしてください。
4.  **出力フォーマット**: あなたは必ず、「curated_memory」（**日本語で**あなたが要約した文字列）と「emotional_valence」（あなたが選んだラベル）の2つのキーを持つ単一のJSONオブジェクトで応答しなければなりません。
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
                emotional_valence=response_json.get("emotional_valence", "中立"),
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
