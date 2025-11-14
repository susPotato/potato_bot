import sys
import os
import shutil
from flask import Flask, render_template, request, jsonify

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

        self.char_manager = CharacterManager(PERSONALITY_FILE, KB_EMBEDDINGS_FILE)
        self.memory_manager = EpisodicMemoryManager(MEMORY_FILE)
        self.curator = Curator(self.curator_llm)
        self.reflector = Reflector(self.reflector_llm)
        self.guardrail = Guardrail()
        self.turn_number = self.memory_manager.memories[-1].turn_number if self.memory_manager.memories else 0
        print(f"ターン番号 {self.turn_number} から開始します")

    def get_response(self, user_input: str) -> str:
        """Handles a single turn of the conversation."""
        debug_log = []
        self.turn_number += 1
        debug_log.append(f"--- ターン {self.turn_number} ---")

        # 1. RAG Search
        debug_log.append(f"記憶を検索中: '{user_input[:30]}...'")
        query_embedding = MODELS.embedding_model.encode(user_input).tolist()
        relevant_memories = self.memory_manager.search_memories(query_embedding, top_k=3)
        debug_log.append(f"{len(relevant_memories)}件の関連する記憶が見つかりました。")
        if relevant_memories:
            for mem in relevant_memories:
                debug_log.append(f"  - 記憶: '{mem.curated_memory[:40]}...'")

        # 2. Prompt Construction
        debug_log.append("メインLLMのプロンプトを構築中。")
        system_prompt = self.char_manager.get_full_persona_text()
        user_prompt = f"これが現在の会話です。\nユーザー: {user_input}\n"
        if relevant_memories:
            user_prompt += "\nこれは過去の会話からの関連する記憶です:\n"
            for mem in relevant_memories:
                user_prompt += f"- {mem.curated_memory}\n"
    
        # 3. Step 1: Generate Internal Monologue
        debug_log.append("ステップ1: 内部的な独白を生成中...")
        monologue_prompt = user_prompt + f"""
さて、{self.char_manager.persona.character.name}として、ユーザーの発言に対して、あなたのペルソナとして、心の中で何を考えているか、何を感じているかを記述してください（日本語で）。
あなたの応答は必ず次の構造を持つJSONオブジェクトでなければなりません:
{{
    "internal_monologue": "あなたの内部的な思考や感情を記述してください（日本語で）。"
}}
"""
        monologue_json = self.main_llm.call(system_prompt, monologue_prompt)
        internal_monologue = monologue_json.get("internal_monologue", "")
        debug_log.append(f"内部的な独白: {internal_monologue[:50]}...")

        # 4. Step 2: Generate Response Based on Monologue
        debug_log.append("ステップ2: 内部的な独白に基づいて応答を生成中...")
        response_prompt = user_prompt + f"""
あなたの内部的な思考:
{internal_monologue}

さて、{self.char_manager.persona.character.name}として、上記の「内部的な思考」で考えたことを、そのまま会話として口に出して話すように表現してください。
重要な点:
- 内部的な思考の具体的な内容（例：「思ってないのかもしれない」「最近の私にはちょっと暗」）を保持してください。
- ただし、それを自然な口語で表現してください（「思ってないのかもしれない」→「思ってないかも」のように）。
- 思考を簡略化したり、一般的な表現に置き換えたりしないでください。
- あなたの思考の流れや感情の変化を反映させてください。

あなたの応答は必ず次の構造を持つJSONオブジェクトでなければなりません:
{{
    "response_message": "あなたの応答メッセージ（日本語で）。"
}}
"""
        bot_response_json = self.main_llm.call(system_prompt, response_prompt)
        debug_log.append(f"LLM 生JSON: {bot_response_json}")
        
        # Extract the message
        try:
            bot_message = bot_response_json.get("response_message", "えっと…なんて言ったらいいか…")
        except (AttributeError, KeyError):
             bot_message = "えっと…なんて言ったらいいか…"

        # 5. Guardrail Check
        debug_log.append("ガードレールで応答をチェック中。")
        _, final_message = self.guardrail.check(bot_message)
        
        # 6. Post-Response Curation
        debug_log.append("このターンの新しい記憶をキュレート中。")
        current_turn = [
            ConversationTurn(speaker="User", message=user_input),
            ConversationTurn(speaker="Potato", message=final_message)
        ]
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
