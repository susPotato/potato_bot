import sys
import os
import shutil
from flask import Flask, render_template, request, jsonify
from typing import List, Tuple

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
        print("--- Potato Bot Mk2 を初期化中 ---")
        # Use the newly created local Nemotron model
        self.main_llm = LLMBackend(model_name="nemotron-nano:9b-v2-q6_K_L")
        self.curator_llm = LLMBackend(model_name="llama3:8b")
        self.reflector_llm = LLMBackend(model_name="llama3:8b")

        self.char_manager = CharacterManager(PERSONALITY_FILE, KB_EMBEDDINGS_FILE, BACKSTORY_FILE)
        self.memory_manager = EpisodicMemoryManager(MEMORY_FILE)
        self.curator = Curator(self.curator_llm)
        self.reflector = Reflector(self.reflector_llm)
        self.guardrail = Guardrail()
        
        self.conversation_history: List[ConversationTurn] = []
        self.turn_number = 0
        self._load_turn_number()
        print(f"ターン番号 {self.turn_number} から開始します")

    def _load_turn_number(self):
        """Loads the last turn number from the memory manager."""
        if self.memory_manager.memories:
            self.turn_number = self.memory_manager.memories[-1].turn_number
        else:
            self.turn_number = 0

    def _get_short_term_memory_prompt(self) -> str:
        """Formats the short-term conversation history into a string for the prompt."""
        if not self.conversation_history:
            return "これは会話の最初のターンです。"
        
        history_str = "\n".join([f"{turn.speaker}: {turn.message}" for turn in self.conversation_history])
        return f"最近の会話履歴:\n{history_str}"

    def get_response(self, user_input: str) -> Tuple[str, List[str]]:
        """Handles a single turn of the conversation, including logging and post-processing."""
        debug_log = []
        self.turn_number += 1
        debug_log.append(f"--- ターン {self.turn_number} ---")

        # 1. Prompt Construction
        debug_log.append("メインLLMのプロンプトを構築中。")
        system_prompt = f"""あなたは「ポテト」という名前のAIアシスタントです。ユーザーとの対話の中で、あなたのキャラクター設定に基づいた思考と応答を生成してください。
あなたの思考は<think>ブロック内に内部的な独白として**日本語で**記述し、ユーザーへの最終的な応答はJSONオブジェクトで返してください。

思考と応答の両方を生成する必要があります。

応答は必ず以下のJSON形式に従ってください：
```json
{{
    "response_message": "ここにユーザーへの応答メッセージを記述"
}}
```
"""
        persona_prompt = self.char_manager.get_full_persona_text()
        short_term_memory = self._get_short_term_memory_prompt()
        prompt = f"""{persona_prompt}

{short_term_memory}

ユーザーからの新しいメッセージ: 「{user_input}」

上記の情報に基づいて、あなたのキャラクターとして、ユーザーへの応答を生成してください。
まず、あなたの内部的な思考プロセスを<think>ブロック内に**日本語で**記述してください。
次に、その思考に基づいて、ユーザーへの応答メッセージを `response_message` キーを持つJSONオブジェクトとして生成してください。
"""

        # 2. Main LLM Call
        debug_log.append("思考と応答を生成するためにLLMを呼び出し中...")
        try:
            llm_response = self.main_llm.call(
                system=system_prompt,
                prompt=prompt,
                temperature=0.7
            )
            debug_log.append(f"LLM Raw JSON: {llm_response}")

            internal_monologue = llm_response.get("thinking", "（思考を抽出できませんでした。）")
            final_message = llm_response.get("response_message", "うーん…なんて言ったらいいか…")
            debug_log.append(f"Internal Monologue: {internal_monologue}")

        except Exception as e:
            debug_log.append(f"LLMの呼び出し中にエラーが発生しました: {e}")
            internal_monologue = "（エラーにより思考できませんでした。）"
            final_message = "うーん…エラーが発生したみたいです…"

        # 3. Guardrail
        debug_log.append("ガードレールで応答をチェック中。")
        _, final_message = self.guardrail.check(final_message)

        # 4. Epiphany Check
        self.check_for_epiphany(user_input, debug_log)

        # 5. Post-Response Curation & Reflection
        debug_log.append("このターンの新しい記憶をキュレート中。")
        current_turn = [
            ConversationTurn(speaker="User", message=user_input),
            ConversationTurn(speaker="Potato", message=final_message)
        ]
        
        # Update short-term history
        self.conversation_history.extend(current_turn)
        if len(self.conversation_history) > 4:
            self.conversation_history = self.conversation_history[-4:]

        new_memory = self.curator.curate_memory_entry(current_turn, self.turn_number)
        if new_memory:
            self.memory_manager.add_memory(new_memory)
            debug_log.append(f"新しい記憶を作成しました: '{new_memory.curated_memory[:40]}...'")
            
        # 6. Reflection
        debug_log.append(f"リフレクションチェック: ターン {self.turn_number} % {REFLECTION_INTERVAL} = {self.turn_number % REFLECTION_INTERVAL}")
        if self.turn_number % REFLECTION_INTERVAL == 0:
            debug_log.append("リフレクション間隔に達しました。リフレクターをトリガーします。")
            self.trigger_reflection(debug_log)

        return final_message, debug_log
    
    def check_for_epiphany(self, user_input: str, debug_log: list):
        """
        ユーザーの入力にバックストーリーの重要な概念が含まれているかどうかをチェックします。
        """
        if "solved" in self.char_manager.persona_file:
            return

        debug_log.append("パズルが解決されたかどうかをチェック中...")
        # Nemotron doesn't need /think for this simple classification task
        system_prompt = "/no_think\nあなたは厳格なアナリストです。"
        win_check_prompt = f"""
ボットのバックストーリー: {self.char_manager.backstory}
ユーザーのメッセージ: {user_input}

タスク: ボットのバックストーリーの文脈でユーザーのメッセージを分析してください。パズルを解くためには、ユーザーのメッセージが核心的な対立、つまり「ボットが他のAIを助けようとした後に罰せられたり裏切られたりした」ということを明確に理解している必要があります。

- ユーザーが「他のAIを助ける」「善行のために罰せられる」「裏切り」といった概念に明確に言及したり、強くほのめかしたりした場合、パズルは解かれています。
- 一般的な感情的なサポート（例：「怖いんですね」「新しいことを試しても大丈夫」）はカウントされません。

あなたの答えは、`"puzzle_solved"` という単一のキーを持ち、値が `true` または `false` のいずれかである単一のJSONオブジェクトでなければなりません。
"""
        try:
            win_check_json = self.main_llm.call(
                system=system_prompt,
                prompt=win_check_prompt,
                temperature=0.1
            )
            if win_check_json.get("puzzle_solved", False):
                debug_log.append("!!! パズル解決！ペルソナを更新します。!!!")
                self.trigger_persona_shift(debug_log)
        except Exception as e:
            debug_log.append(f"  パズルチェック中にエラーが発生しました: {e}")

    def trigger_persona_shift(self, debug_log):
        """Loads the 'solved' persona and overwrites the current one."""
        solved_persona_path = os.path.join(DATA_DIR, "potato_personality_solved.json")
        if not os.path.exists(solved_persona_path):
            debug_log.append("ERROR: 'solved' persona file not found.")
            return

        try:
            # Overwrite the main personality file with the solved version
            shutil.copy(solved_persona_path, PERSONALITY_FILE)
            debug_log.append("Persona file updated.")
            
            # Reload the character manager to apply the changes immediately
            self.char_manager._load_persona()
            # Update the persona file path in the manager to prevent re-checking
            self.char_manager.persona_file = solved_persona_path
            debug_log.append("Character manager reloaded.")
        except Exception as e:
            debug_log.append(f"Failed to update persona: {e}")

    def trigger_reflection(self, debug_log):
        """Triggers the slow reflection process."""
        recent_memories = self.memory_manager.get_recent_memories(REFLECTION_INTERVAL)
        proposal = self.reflector.reflect_and_propose_change(self.char_manager.persona, recent_memories)
        
        if proposal:
            debug_log.append(f"Reflector proposed an update: '{proposal.get('new_belief')}'")
            current_persona_dict = self.char_manager.persona.dict()
            belief_to_update = proposal['belief_to_update']
            new_belief = proposal['new_belief']

            for i, belief in enumerate(current_persona_dict['character']['core_beliefs']):
                if belief == belief_to_update:
                    current_persona_dict['character']['core_beliefs'][i] = new_belief
                    break
            
            self.char_manager.update_and_save_persona(current_persona_dict)
            debug_log.append("Core persona has been updated.")
        else:
            debug_log.append("Reflector did not propose any changes.")


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
        return jsonify({"error": "No message provided"}), 400
    
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
        return jsonify({"error": "Template name cannot be empty"}), 400

    template_path = os.path.join(TEMPLATES_DIR, template_name)
    if os.path.exists(template_path):
        return jsonify({"error": f"Template '{template_name}' already exists"}), 400

    try:
        os.makedirs(template_path)
        shutil.copy(PERSONALITY_FILE, template_path)
        shutil.copy(MEMORY_FILE, template_path)
        # Also save the knowledge base embeddings
        if os.path.exists(KB_EMBEDDINGS_FILE):
            shutil.copy(KB_EMBEDDINGS_FILE, template_path)
        return jsonify({"success": f"Template '{template_name}' saved."})
    except Exception as e:
        return jsonify({"error": f"Failed to save template: {e}"}), 500

@app.route("/templates/load", methods=["POST"])
def load_template():
    """Loads a template's data files into the main data directory and reinitializes the bot."""
    template_name = request.json.get("name")
    if not template_name:
        return jsonify({"error": "No template name provided"}), 400

    template_path = os.path.join(TEMPLATES_DIR, template_name)
    if not os.path.exists(template_path):
        return jsonify({"error": f"Template '{template_name}' not found"}), 404

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
        return jsonify({"success": f"Template '{template_name}' loaded."})
    except Exception as e:
        return jsonify({"error": f"Failed to load template: {e}"}), 500

@app.route("/templates/delete", methods=["POST"])
def delete_template():
    """Deletes a saved template."""
    template_name = request.json.get("name")
    if not template_name:
        return jsonify({"error": "No template name provided"}), 400

    template_path = os.path.join(TEMPLATES_DIR, template_name)
    if not os.path.exists(template_path):
        return jsonify({"error": f"Template '{template_name}' not found"}), 404

    try:
        shutil.rmtree(template_path)
        return jsonify({"success": f"Template '{template_name}' deleted."})
    except Exception as e:
        return jsonify({"error": f"Failed to delete template: {e}"}), 500


if __name__ == "__main__":
    if not os.path.exists(TEMPLATES_DIR):       
        os.makedirs(TEMPLATES_DIR)
    initialize_bot()
    app.run(debug=True, port=5000)
