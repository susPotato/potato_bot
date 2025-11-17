from transformers import MarianMTModel, MarianTokenizer

class Translator:
    """A class to handle Japanese-English and English-Japanese translation."""
    def __init__(self):
        print("--- Initializing Translation Models ---")
        try:
            # Load Japanese to English model
            self.ja_en_model_name = 'Helsinki-NLP/opus-mt-ja-en'
            self.ja_en_tokenizer = MarianTokenizer.from_pretrained(self.ja_en_model_name)
            self.ja_en_model = MarianMTModel.from_pretrained(self.ja_en_model_name, use_safetensors=True)
            print("  - Japanese to English model loaded.")

            # Load English to Japanese model - SWITCHING TO FUGUMT
            self.en_ja_model_name = 'staka/fugumt-en-ja' 
            self.en_ja_tokenizer = MarianTokenizer.from_pretrained(self.en_ja_model_name)
            self.en_ja_model = MarianMTModel.from_pretrained(self.en_ja_model_name, use_safetensors=True)
            print("  - English to Japanese model loaded.")
            print("--- Translation Models Initialized Successfully ---")
        except Exception as e:
            print(f"--- ❌ ERROR: Failed to initialize translation models: {e} ---")
            print("--- Please ensure you have a stable internet connection for the first-time download. ---")
            raise

    def ja_to_en(self, text: str) -> str:
        """Translates Japanese text to English."""
        try:
            batch = self.ja_en_tokenizer([text], return_tensors="pt")
            gen = self.ja_en_model.generate(**batch)
            return self.ja_en_tokenizer.decode(gen[0], skip_special_tokens=True)
        except Exception as e:
            print(f"Error during Ja->En translation: {e}")
            return "[Translation Error]"

    def en_to_ja(self, text: str) -> str:
        """Translates English text to Japanese."""
        try:
            batch = self.en_ja_tokenizer([text], return_tensors="pt")
            gen = self.en_ja_model.generate(**batch)
            return self.en_ja_tokenizer.decode(gen[0], skip_special_tokens=True)
        except Exception as e:
            print(f"Error during En->Ja translation: {e}")
            return "[Translation Error]"

# Global instance to be used by the app
TRANSLATOR = Translator()
