import ollama
import traceback

class LLMBackend:
    def __init__(self, model_name="llama3:8b"):
        self.model = model_name

    def call(self, system: str, prompt: str, temperature=0.8):
        """
        Calls the local Ollama model with a system message and a user prompt.
        """
        try:
            response = ollama.chat(
                model=self.model,
                messages=[
                    {'role': 'system', 'content': system},
                    {'role': 'user', 'content': prompt},
                ],
                options={
                    "temperature": temperature,
                }
            )
            return response['message']['content']
        except Exception as e:
            print(f"❌ Error in LLM call: {str(e)}")
            traceback.print_exc()
            # Return a structured error message that the main loop can handle
            return '{"error": "Failed to get response from LLM."}'
