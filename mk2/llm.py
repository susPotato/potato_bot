from ollama import Client
import json

class LLMBackend:
    """A simple wrapper for the Ollama API client."""
    def __init__(self, model_name: str, host: str = "http://localhost:11434"):
        self.client = Client(host=host)
        self.model_name = model_name

    def call(self, system: str, prompt: str, temperature: float = 0.5) -> dict:
        """Makes a call to the LLM and returns the parsed JSON response including any thinking."""
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
            
            # Robust JSON parsing from the main content
            content = response['message']['content']
            parsed_json = {}
            try:
                json_start = content.find('{')
                json_end = content.rfind('}') + 1
                if json_start != -1 and json_end != -1:
                    json_str = content[json_start:json_end]
                    parsed_json = json.loads(json_str)
                else:
                    # Handle cases where no JSON is found
                    parsed_json = {"error": "No JSON object found in the response"}
            except json.JSONDecodeError:
                 parsed_json = {"error": "Failed to decode JSON from the response"}


            # Add the 'thinking' part to our final dictionary if it exists
            if 'thinking' in response['message']:
                parsed_json['thinking'] = response['message']['thinking']

            return parsed_json
        
        except Exception as e:
            print(f"An error occurred in LLMBackend: {e}")
            return {"error": str(e)}
