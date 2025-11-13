from chat import ChatManager
from character import Character
from llm import LLMBackend
import os

# --- Important ---
# Make sure you have Ollama installed and running.
# Pull the models you need:
# `ollama pull llama3:8b`
# `ollama pull nomic-embed-text`
# -----------------

def main():
    print("Initializing Potato Bot...")

    # Define file paths
    # We expect these files to be in the same directory as main.py
    script_dir = os.path.dirname(os.path.realpath(__file__))
    personality_file = os.path.join(script_dir, "potato_personality.json")
    chat_history_file = os.path.join(script_dir, "potato_chat.json")

    # Initialize components
    try:
        chat_manager = ChatManager(chat_save_file=chat_history_file)
        character = Character(character_file=personality_file)
        llm = LLMBackend(model_name="llama3:8b") # Or whatever model you prefer
    except FileNotFoundError as e:
        print(f"Error: A required file was not found: {e}")
        print("Please make sure potato_personality.json and potato_chat.json are in the same directory.")
        return
    except Exception as e:
        print(f"An unexpected error occurred during initialization: {e}")
        return

    print("--- Potato Bot is Ready ---")
    print("Type 'quit' or 'exit' to end the chat.")
    print("\n")

    # Get the character's name for the prompt
    bot_name = character.name

    while True:
        # Get user input
        user_input = input("You: ")

        if user_input.lower() in ["quit", "exit"]:
            print("Potato: Goodbye!")
            break
        
        # Add user message to chat history
        # Here, we'll just use a generic user name. 
        # You could modify this to ask for a name at the start.
        chat_manager.add_chat(author="User", message=user_input)

        # Render the prompt for the LLM
        system_prompt, user_prompt = character.render_prompt(chat_manager.get_chat_list())

        # Call the local LLM
        response_json = llm.call(system=system_prompt, prompt=user_prompt)

        # Process the response
        if response_json:
            try:
                # Update character's knowledge and memories
                character.update_info(response_json)

                # If the bot decides to respond, print the message
                if response_json.get("should_respond"):
                    bot_message = response_json.get("response_message", "I'm not sure what to say.")
                    print(f"{bot_name}: {bot_message}")

                    # Add bot's response to the chat history
                    chat_manager.add_chat(author=bot_name, message=bot_message)
                else:
                    print(f"({bot_name} chose not to respond.)")
                
                print("\n") # Add a little space for readability

            except Exception as e:
                print(f"An error occurred while processing the response: {e}")
                print("The response from the model might have been malformed.")

if __name__ == "__main__":
    main()
