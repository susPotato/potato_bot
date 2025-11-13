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
        print("--- Initializing Potato Bot Mk1 ---")
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
        print(f"Starting at turn number {self.turn_number}")

    def get_response(self, user_input: str) -> str:
        """Handles a single turn of the conversation."""
        debug_log = []
        self.turn_number += 1
        debug_log.append(f"--- Turn {self.turn_number} ---")

        # 1. RAG Search
        debug_log.append(f"Searching memories for: '{user_input[:30]}...'")
        query_embedding = MODELS.embedding_model.encode(user_input).tolist()
        relevant_memories = self.memory_manager.search_memories(query_embedding, top_k=3)
        debug_log.append(f"Found {len(relevant_memories)} relevant memories.")
        if relevant_memories:
            for mem in relevant_memories:
                debug_log.append(f"  - Memory: '{mem.curated_memory[:40]}...'")

        # 2. Prompt Construction
        debug_log.append("Constructing prompt for main LLM.")
        system_prompt = self.char_manager.get_full_persona_text()
        user_prompt = f"This is the current conversation.\nUser: {user_input}\n"
        if relevant_memories:
            user_prompt += "\nHere are some relevant memories from our past conversations:\n"
            for mem in relevant_memories:
                user_prompt += f"- {mem.curated_memory}\n"
    
        user_prompt += f"""
Now, as {self.char_manager.persona.character.name}, your response MUST be a JSON object with the following structure:
{{
    "thoughts": "Your brief thought process on how to respond to the user, keeping your persona in mind.",
    "response_message": "The message you want to say to the user."
}}
"""

        # 3. Get Response
        debug_log.append("Calling main LLM for response...")
        bot_response_json = self.main_llm.call(system_prompt, user_prompt)
        debug_log.append(f"LLM Raw JSON: {bot_response_json}")
        bot_message = bot_response_json.get("response_message", "I... I'm not sure what to say.")

        # 4. Guardrail Check
        debug_log.append("Checking response against guardrail.")
        _, final_message = self.guardrail.check(bot_message)
        
        # 5. Post-Response Curation
        debug_log.append("Curating new memory for this turn.")
        current_turn = [
            ConversationTurn(speaker="User", message=user_input),
            ConversationTurn(speaker="Potato", message=final_message)
        ]
        new_memory = self.curator.curate_memory_entry(current_turn, self.turn_number)
        if new_memory:
            self.memory_manager.add_memory(new_memory)
            debug_log.append(f"New memory created: '{new_memory.curated_memory[:40]}...'")
            
        # 6. Reflection
        debug_log.append(f"Reflection check: Turn {self.turn_number} % {REFLECTION_INTERVAL} = {self.turn_number % REFLECTION_INTERVAL}")
        if self.turn_number % REFLECTION_INTERVAL == 0:
            debug_log.append("Reflection interval reached. Triggering reflector.")
            self.trigger_reflection(debug_log)

        return final_message, debug_log

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
        # If it doesn't exist in the template, we should remove any existing one in data to force recreation
        elif os.path.exists(KB_EMBEDDINGS_FILE):
            os.remove(KB_EMBEDDINGS_FILE)

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
