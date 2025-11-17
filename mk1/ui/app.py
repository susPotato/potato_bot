import sys
import os
import shutil
from flask import Flask, render_template, request, jsonify
from typing import List

# Add the parent 'mk1' directory to the Python path to find our modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from character_manager import CharacterManager
from episodic_memory_manager import EpisodicMemoryManager
from curator import Curator
from reflector import Reflector
from guardrail import Guardrail
from llm import LLMBackend
from models import MODELS
from schemas import ConversationTurn

# --- Constants ---
# Construct absolute paths based on the location of this script
_script_dir = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(_script_dir, "..", "data")
TEMPLATES_DIR = os.path.join(_script_dir, "..", "templates")
PERSONALITY_FILE = os.path.join(DATA_DIR, "potato_personality.json")
MEMORY_FILE = os.path.join(DATA_DIR, "episodic_memory.json")
BACKSTORY_FILE = os.path.join(DATA_DIR, "backstory.txt")
KB_EMBEDDINGS_FILE = os.path.join(DATA_DIR, "kb_embeddings.pkl")
REFLECTION_INTERVAL = 10 

class PotatoBot:
    """A class to encapsulate the entire bot's functionality."""
    def __init__(self):
        print("--- Potato Bot Mk1 を初期化中 ---")
        # We might use different models for different tasks
        self.main_llm = LLMBackend(model_name="llama3:8b")
        self.curator_llm = LLMBackend(model_name="llama3:8b")
        self.reflector_llm = LLMBackend(model_name="llama3:8b")

        self.char_manager = CharacterManager(PERSONALITY_FILE, KB_EMBEDDINGS_FILE, BACKSTORY_FILE)
        self.memory_manager = EpisodicMemoryManager(MEMORY_FILE)
        self.curator = Curator(self.curator_llm)
        self.reflector = Reflector(self.reflector_llm)
        self.guardrail = Guardrail()
        self.turn_number = self.memory_manager.memories[-1].turn_number if self.memory_manager.memories else 0
        self.conversation_history: List[ConversationTurn] = []
        print(f"ターン番号 {self.turn_number} から開始します")

    def get_response(self, user_input: str) -> str:
        """Handles a single turn of the conversation."""
        debug_log = []
        self.turn_number += 1
        debug_log.append(f"--- ターン {self.turn_number} ---")

        # RAG is disabled to focus on the core story
        debug_log.append("RAG検索は無効化されています。")
        
        # 1. Prompt Construction
        debug_log.append("メインLLMのプロンプトを構築中。")
        system_prompt = self.char_manager.get_full_persona_text()
        history_str = "\n".join([f"{turn.speaker}: {turn.message}" for turn in self.conversation_history])

        # --- STEP 1: GENERATE INTERNAL MONOLOGUE ---
        debug_log.append("ステップ1: 内部的な独白を生成中...")
        monologue_prompt = f"""
直近の会話履歴:
---
{history_str}
---
ユーザーからの最新のメッセージ: "{user_input}"

あなたの秘密のバックストーリー:
---
{self.char_manager.backstory}
---

タスク: 上記の情報に基づいて、あなたのペルソナとして、心の中で何を考えているか、何を感じているかを記述してください。

あなたの応答は、次のキーを持つJSONオブジェクトでなければなりません:
{{
    "internal_monologue": "あなたの内部的な思考や感情（日本語で）"
}}
"""
        monologue_json = self.main_llm.call(system_prompt, monologue_prompt, temperature=0.3)
        internal_monologue = monologue_json.get("internal_monologue", "（独白の生成に失敗しました）")
        debug_log.append(f"内部的な独白: {internal_monologue[:80]}...")

        # --- STEP 2: GENERATE HINT FROM MONOLOGUE ---
        debug_log.append("ステップ2: 独白からヒントを生成中...")
        hint_prompt = f"""
あなたのキャラクターの内部的な思考:
---
{internal_monologue}
---

あなたのタスク:
上記の「内部的な思考」を、ユーザーへの**巧妙なヒント**に変換してください。

重要なルール:
- **応答は必ず日本語でなければなりません。**
- バックストーリーの事実を**直接**話してはいけません。
- あなたのヒントは、思考の中で言及されている**具体的な出来事**（例：「別のAIを助けたこと」「罰せられたこと」）を**間接的に**示唆するものでなければなりません。
- 応答は、あなたのキャラクターの躊躇いや悲しみが伝わるような、自然な会話でなければなりません。

あなたの応答は、次のキーを持つJSONオブジェクトでなければなりません:
{{
    "response_message": "ユーザーへの実際の応答（巧妙なヒント）"
}}
"""
        bot_response_json = self.main_llm.call(system_prompt, hint_prompt, temperature=0.4)
        debug_log.append(f"LLM 生JSON: {bot_response_json}")
        
        # Extract the message
        try:
            bot_message = bot_response_json.get("response_message", "えっと…なんて言ったらいいか…")
        except (AttributeError, KeyError):
             bot_message = "えっと…なんて言ったらいいか…"

        # 3. Guardrail Check
        debug_log.append("ガードレールで応答をチェック中。")
        _, final_message = self.guardrail.check(bot_message)
        
        # New Step: Check if the user solved the puzzle
        self.check_for_epiphany(user_input, debug_log)

        # 4. Post-Response Curation & Reflection
        debug_log.append("このターンの新しい記憶をキュレート中。")
        current_turn = [
            ConversationTurn(speaker="User", message=user_input),
            ConversationTurn(speaker="Potato", message=final_message)
        ]
        
        # Update short-term history
        self.conversation_history.extend(current_turn)
        # Keep the history to the last 4 turns (2 user, 2 bot)
        if len(self.conversation_history) > 4:
            self.conversation_history = self.conversation_history[-4:]

        new_memory = self.curator.curate_memory_entry(current_turn, self.turn_number)
        if new_memory:
            self.memory_manager.add_memory(new_memory)
            debug_log.append(f"新しい記憶を作成しました: '{new_memory.curated_memory[:40]}...'")
            
        # 7. Reflection
        debug_log.append(f"リフレクションチェック: ターン {self.turn_number} % {REFLECTION_INTERVAL} = {self.turn_number % REFLECTION_INTERVAL}")
        if self.turn_number % REFLECTION_INTERVAL == 0:
            debug_log.append("リフレクションの間隔に達しました。リフレクターを起動します。")
            self.trigger_reflection(debug_log)

        return final_message, debug_log

    def check_for_epiphany(self, user_input: str, debug_log: list):
        """Checks if the user's input solves the bot's backstory puzzle."""
        # Don't check if the persona has already been solved
        if "solved" in self.char_manager.persona_file:
            return

        debug_log.append("パズルが解かれたかチェック中...")
        win_check_prompt = f"""
Bot's Backstory: {self.char_manager.backstory}
User's Message: {user_input}

Task: Analyze the user's message in the context of the bot's backstory. To solve the puzzle, the user's message MUST demonstrate a clear understanding of the core conflict: that the bot was punished or betrayed after trying to HELP another AI.

- If the user explicitly mentions or strongly alludes to concepts like "helping another AI," "being punished for a good deed," or "betrayal," then the puzzle is solved.
- General emotional support (e.g., "I understand you're scared," "It's okay to try new things") does NOT count.

Your answer must be a single JSON object with one key, "puzzle_solved", set to either true or false.
"""
        try:
            win_check_json = self.main_llm.call(
                system="You are a strict analyst.", # Use a neutral system prompt for this task
                prompt=win_check_prompt,
                temperature=0.1
            )
            if win_check_json.get("puzzle_solved", False):
                debug_log.append("！！！パズルが解かれました！ペルソナを更新します。！！！")
                self.trigger_persona_shift(debug_log)
        except Exception as e:
            debug_log.append(f"  パズルチェック中にエラーが発生しました: {e}")

    def trigger_persona_shift(self, debug_log):
        """Loads the 'solved' persona and overwrites the current one."""
        solved_persona_path = os.path.join(DATA_DIR, "potato_personality_solved.json")
        if not os.path.exists(solved_persona_path):
            debug_log.append("エラー: 'solved'ペルソナファイルが見つかりません。")
            return

        try:
            # Overwrite the main personality file with the solved version
            shutil.copy(solved_persona_path, PERSONALITY_FILE)
            debug_log.append("ペルソナファイルを更新しました。")
            
            # Reload the character manager to apply the changes immediately
            self.char_manager._load_persona()
            # Update the persona file path in the manager to prevent re-checking
            self.char_manager.persona_file = solved_persona_path
            debug_log.append("キャラクターマネージャーをリロードしました。")
        except Exception as e:
            debug_log.append(f"ペルソナの更新に失敗しました: {e}")

    def trigger_reflection(self, debug_log):
        """Triggers the slow reflection process."""
        recent_memories = self.memory_manager.get_recent_memories(REFLECTION_INTERVAL)
        proposal = self.reflector.reflect_and_propose_change(self.char_manager.persona, recent_memories)
        
        if proposal:
            debug_log.append(f"リフレクターが更新を提案しました: '{proposal.get('new_belief')}'")
            current_persona_dict = self.char_manager.persona.dict()
            belief_to_update = proposal['belief_to_update']
            new_belief = proposal['new_belief']

            for i, belief in enumerate(current_persona_dict['character']['core_beliefs']):
                if belief == belief_to_update:
                    current_persona_dict['character']['core_beliefs'][i] = new_belief
                    break
            
            self.char_manager.update_and_save_persona(current_persona_dict)
            debug_log.append("コアペルソナが更新されました。")
        else:
            debug_log.append("リフレクターは変更を提案しませんでした。")


# --- Flask App ---
app = Flask(__name__)
bot = None # Bot will be initialized after ensuring files are in place

def initialize_bot():
    """Initializes or re-initializes the global bot instance."""
    global bot
    bot = PotatoBot()

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat():
    user_message = request.json.get("message")
    if not user_message:
        return jsonify({"error": "メッセージがありません"}), 400
    
    bot_response, debug_log = bot.get_response(user_message)
    return jsonify({"response": bot_response, "debug_log": debug_log})

@app.route("/templates", methods=["GET"])
def get_templates():
    """Returns a list of available template names."""
    if not os.path.exists(TEMPLATES_DIR):
        return jsonify([])
    templates = [d for d in os.listdir(TEMPLATES_DIR) if os.path.isdir(os.path.join(TEMPLATES_DIR, d))]
    return jsonify(sorted(templates))

@app.route("/templates/save", methods=["POST"])
def save_template():
    """Saves the current data files as a new template."""
    template_name = request.json.get("name")
    if not template_name or not template_name.strip():
        return jsonify({"error": "テンプレート名は空にできません"}), 400

    template_path = os.path.join(TEMPLATES_DIR, template_name)
    if os.path.exists(template_path):
        return jsonify({"error": f"テンプレート「{template_name}」は既に存在します"}), 400

    try:
        os.makedirs(template_path)
        shutil.copy(PERSONALITY_FILE, template_path)
        shutil.copy(MEMORY_FILE, template_path)
        # Also save the knowledge base embeddings
        if os.path.exists(KB_EMBEDDINGS_FILE):
            shutil.copy(KB_EMBEDDINGS_FILE, template_path)
        return jsonify({"success": f"テンプレート「{template_name}」を保存しました。"})
    except Exception as e:
        return jsonify({"error": f"テンプレートの保存に失敗しました: {e}"}), 500

@app.route("/templates/load", methods=["POST"])
def load_template():
    """Loads a template's data files into the main data directory and reinitializes the bot."""
    template_name = request.json.get("name")
    if not template_name:
        return jsonify({"error": "テンプレート名がありません"}), 400

    template_path = os.path.join(TEMPLATES_DIR, template_name)
    if not os.path.exists(template_path):
        return jsonify({"error": f"テンプレート「{template_name}」が見つかりません"}), 404

    try:
        # --- Explicitly delete current data files to ensure a clean slate ---
        if os.path.exists(PERSONALITY_FILE):
            os.remove(PERSONALITY_FILE)
        if os.path.exists(MEMORY_FILE):
            os.remove(MEMORY_FILE)
        if os.path.exists(KB_EMBEDDINGS_FILE):
            os.remove(KB_EMBEDDINGS_FILE)

        # Define source paths for all files in the template
        template_personality_file = os.path.join(template_path, os.path.basename(PERSONALITY_FILE))
        template_memory_file = os.path.join(template_path, os.path.basename(MEMORY_FILE))
        template_embeddings_file = os.path.join(template_path, os.path.basename(KB_EMBEDDINGS_FILE))

        # Copy all files from the template to the data directory
        shutil.copy(template_personality_file, DATA_DIR)
        shutil.copy(template_memory_file, DATA_DIR)
        
        # The embeddings file might not exist in older templates, so copy only if it's there
        if os.path.exists(template_embeddings_file):
            shutil.copy(template_embeddings_file, DATA_DIR)

        initialize_bot() # Re-initialize the bot with the new data
        return jsonify({"success": f"テンプレート「{template_name}」を読み込みました。"})
    except Exception as e:
        return jsonify({"error": f"テンプレートの読み込みに失敗しました: {e}"}), 500

@app.route("/templates/delete", methods=["POST"])
def delete_template():
    """Deletes a saved template."""
    template_name = request.json.get("name")
    if not template_name:
        return jsonify({"error": "テンプレート名がありません"}), 400

    template_path = os.path.join(TEMPLATES_DIR, template_name)
    if not os.path.exists(template_path):
        return jsonify({"error": f"テンプレート「{template_name}」が見つかりません"}), 404

    try:
        shutil.rmtree(template_path)
        return jsonify({"success": f"テンプレート「{template_name}」を削除しました。"})
    except Exception as e:
        return jsonify({"error": f"テンプレートの削除に失敗しました: {e}"}), 500


if __name__ == "__main__":
    if not os.path.exists(TEMPLATES_DIR):       
        os.makedirs(TEMPLATES_DIR)
    initialize_bot()
    app.run(debug=True, port=5000)
