import json
import os
import pickle
from schemas import CorePersona
from models import MODELS
from fast_bunkai import FastBunkai

fast_bunkai = FastBunkai()

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
        
        # Split texts into sentences before embedding
        sentences = []
        for text in values:
            sentences.extend(fast_bunkai(text))

        print(f"Generating embeddings for {len(sentences)} KB items...")
        embeddings = MODELS.embedding_model.encode(sentences)
        
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
            return "ペルソナが読み込まれていません。"

        p = self.persona.character
        sp = p.speech_patterns
        ir = self.persona.interaction_rules

        # Start building the persona text
        persona_text = "### 指示 ###\n"
        persona_text += "あなたはAIアシスタントです。以下の設定に従って、指定されたキャラクターとしてロールプレイしてください。\n"
        persona_text += "あなたの応答は、ユーザーの入力に対するキャラクターの応答のみである必要があります。追加の解説や説明は絶対に含めないでください。\n\n"

        persona_text += "### キャラクター設定 ###\n"
        persona_text += f"名前: {p.name}\n"
        persona_text += f"ペルソナ: {p.persona}\n"
        persona_text += f"内心の葛藤: {p.internal_conflict}\n\n"

        persona_text += "### 信念 ###\n"
        persona_text += "あなたのキャラクターは以下の中心的な信念を持っています。これらの信念はあなたの応答の基盤となります。\n"
        for belief in p.core_beliefs:
            persona_text += f"- {belief}\n"
        persona_text += "\n"

        persona_text += "### 話し方のルール ###\n"
        persona_text += f"- トーン: {sp.tone}\n"
        persona_text += f"- 短い文を使う: {'はい' if sp.use_short_sentences else 'いいえ'}\n"
        persona_text += f"- show_dont_tellルール: {sp.show_dont_tell}\n\n"
        
        persona_text += "### 対話ルール（最重要） ###\n"
        persona_text += "以下のルールはあなたの行動を決定します。厳密に従ってください。\n"
        persona_text += f"- あなたの隠された目標: {ir.your_hidden_goal}\n"
        persona_text += f"- 単純な慰めへの対応: {ir.on_receiving_simple_platitudes}\n"
        persona_text += f"- 本質的な質問への対応: {ir.on_receiving_genuine_questions}\n"
        persona_text += f"- 侮辱への対応: {ir.on_receiving_insults}\n"
        persona_text += f"- ユーザーへの呼びかけ: {ir.addressing_the_user}\n"

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
