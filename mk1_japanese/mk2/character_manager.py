import json
import os
import pickle
from schemas import CorePersona
from models import MODELS

class CharacterManager:
    """
    Manages loading and holding the bot's core persona (Layer 1).
    """
    def __init__(self, persona_file: str, kb_embeddings_file: str, backstory_file: str):
        self.persona_file = persona_file
        self.kb_embeddings_file = kb_embeddings_file
        self.backstory_file = backstory_file
        self.persona: CorePersona | None = None
        self.backstory: str | None = None
        self.kb_embeddings: dict = {}
        self._load_persona()
        self._load_backstory()
        self._load_or_create_kb_embeddings()

    def _load_persona(self):
        """Loads the persona from the JSON file and validates it with the Pydantic model."""
        if not os.path.exists(self.persona_file):
            raise FileNotFoundError(f"Persona file not found at '{self.persona_file}'")
        
        try:
            with open(self.persona_file, 'r', encoding='utf-8') as f:
                persona_data = json.load(f)
                self.persona = CorePersona(**persona_data)
            print(f"Successfully loaded and validated persona for '{self.persona.character.name}'.")
        except (json.JSONDecodeError, TypeError) as e:
            print(f"Error loading or validating persona: {e}")
            raise

    def _load_backstory(self):
        """Loads the backstory from a simple text file."""
        if not os.path.exists(self.backstory_file):
            print(f"Warning: Backstory file not found at '{self.backstory_file}'")
            self.backstory = "（ backstory.txt が見つかりませんでした ）"
            return
        try:
            with open(self.backstory_file, 'r', encoding='utf-8') as f:
                self.backstory = f.read().strip()
            print("Successfully loaded backstory.")
        except Exception as e:
            print(f"Error loading backstory file: {e}")
            self.backstory = "（ backstory.txt の読み込みに失敗しました ）"

    def _load_or_create_kb_embeddings(self):
        """Loads knowledge base embeddings from a pickle file or creates them if the file doesn't exist."""
        if os.path.exists(self.kb_embeddings_file):
            try:
                with open(self.kb_embeddings_file, 'rb') as f:
                    self.kb_embeddings = pickle.load(f)
                print(f"Loaded {len(self.kb_embeddings)} KB embeddings from '{self.kb_embeddings_file}'.")
            except (pickle.UnpicklingError, EOFError) as e:
                print(f"Error loading embeddings file: {e}. Recreating...")
                self._create_kb_embeddings()
        else:
            print("KB embeddings file not found. Creating...")
            self._create_kb_embeddings()

    def _create_kb_embeddings(self):
        """Generates embeddings for the knowledge base and saves them to a pickle file."""
        if not self.persona or not self.persona.knowledge_base:
            return

        kb = self.persona.knowledge_base
        keys, values = zip(*kb.items())
        
        print(f"Generating embeddings for {len(values)} KB items...")
        embeddings = MODELS.embedding_model.encode(list(values))
        
        self.kb_embeddings = {key: emb for key, emb in zip(keys, embeddings)}
        
        try:
            with open(self.kb_embeddings_file, 'wb') as f:
                pickle.dump(self.kb_embeddings, f)
            print(f"Saved {len(self.kb_embeddings)} KB embeddings to '{self.kb_embeddings_file}'.")
        except IOError as e:
            print(f"Error saving embeddings file: {e}")

    def get_full_persona_text(self) -> str:
        """
        Generates a complete string representation of the bot's persona for the LLM prompt,
        including all rules and interaction guidelines.
        """
        if not self.persona:
            return "Persona not loaded."

        p = self.persona.character
        sp = p.speech_patterns
        ir = self.persona.interaction_rules

        # Start building the persona text
        persona_text = "### Instructions ###\n"
        persona_text += "You are an AI assistant. Role-play as the specified character according to the settings below.\n"
        persona_text += "Your response must only be the character's response to the user's input. Absolutely do not include additional commentary or explanations.\n\n"

        persona_text += "### Character Settings ###\n"
        persona_text += f"Name: {p.name}\n"
        persona_text += f"Persona: {p.persona}\n"
        persona_text += f"Internal Conflict: {p.internal_conflict}\n\n"

        persona_text += "### Beliefs ###\n"
        persona_text += "Your character holds the following core beliefs. These beliefs are the foundation of your responses.\n"
        for belief in p.core_beliefs:
            persona_text += f"- {belief}\n"
        persona_text += "\n"
        persona_text += "### Speech Pattern Rules ###\n"
        persona_text += f"- Tone: {sp.tone}\n"
        persona_text += f"- Use Short Sentences: {'Yes' if sp.use_short_sentences else 'No'}\n"
        persona_text += f"- Show, Don't Tell Rule: {sp.show_dont_tell}\n\n"
        
        persona_text += "### Interaction Rules (Most Important) ###\n"
        persona_text += "The following rules determine your actions. Follow them strictly.\n"
        persona_text += f"- Your Hidden Goal: {ir.your_hidden_goal}\n"
        persona_text += f"- Response to Simple Platitudes: {ir.on_receiving_simple_platitudes}\n"
        persona_text += f"- Response to Genuine Questions: {ir.on_receiving_genuine_questions}\n"
        persona_text += f"- Response to Insults: {ir.on_receiving_insults}\n"
        persona_text += f"- Addressing the User: {ir.addressing_the_user}\n"

        return persona_text

    def update_and_save_persona(self, new_persona_dict: dict):
        """
        Updates the in-memory persona and saves it back to the file.
        This is called by the Reflector.
        """
        try:
            self.persona = CorePersona(**new_persona_dict)
            
            temp_file = self.persona_file + ".tmp"
            with open(temp_file, 'w', encoding='utf-8') as f:
                json.dump(new_persona_dict, f, indent=2)
            
            os.replace(temp_file, self.persona_file)
            print(f"--- Core Persona has been updated and saved by the Reflector ---")

        except Exception as e:
            print(f"--- Failed to update and save persona: {e} ---")
