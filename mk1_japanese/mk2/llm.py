from ollama import Client
import json

class LLMBackend:
    """A simple wrapper for the Ollama API client."""
    def __init__(self, model_name: str, host: str = "http://localhost:11434"):
        self.client = Client(host=host)
        self.model_name = model_name

    def call(self, system: str, prompt: str, temperature: float = 0.5) -> dict:
        """Makes a call to the LLM and returns the parsed JSON response."""
        try:
            response = self.client.chat(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": prompt}
                ],
                format="json",
                options={"temperature": temperature}
            )
            return json.loads(response['message']['content'])
        except Exception as e:
            print(f"Error calling LLM: {e}")
            return {"error": str(e)}
