import ollama
import traceback
import json

class LLMBackend:
    """
    A simple wrapper for making calls to a local Ollama model.
    """
    def __init__(self, model_name="llama3:8b"):
        self.model = model_name

    def call(self, system: str, prompt: str, temperature=0.7) -> dict:
        """
        Calls the local Ollama model and expects a JSON response.
        """
        try:
            response = ollama.chat(
                model=self.model,
                messages=[
                    {'role': 'system', 'content': system},
                    {'role': 'user', 'content': prompt},
                ],
                format="json", # Ollama's JSON mode is very helpful
                options={"temperature": temperature}
            )
            # The response content should be a JSON string
            return json.loads(response['message']['content'])
        except Exception as e:
            print(f"❌ Error in LLM call: {str(e)}")
            traceback.print_exc()
            return {"error": "Failed to get a valid JSON response from LLM."}
