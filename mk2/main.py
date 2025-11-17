import os
from character_manager import CharacterManager
from episodic_memory_manager import EpisodicMemoryManager
from curator import Curator
from reflector import Reflector
from guardrail import Guardrail
from llm import LLMBackend
from models import MODELS
from schemas import ConversationTurn
from sota_socket_interface import SotaSocket
import time

# --- Constants ---
PERSONALITY_FILE = os.path.join("data", "potato_personality.json")
MEMORY_FILE = os.path.join("data", "episodic_memory.json")
REFLECTION_INTERVAL = 10 # Reflect after every 10 turns

def main():
    print("--- Initializing Potato Bot Mk1 ---")

    # --- Initialize Sota Connection ---
    sota_client = SotaSocket(bot_name="MK1_POTATO")
    print("--- Attempting to connect to Sota system... ---")
    if not sota_client.connect():
        print("--- Connection Failed. Please ensure the Sota server is running. ---")
        return

    # --- Initialize Backend Systems ---
    try:
        # We might use different models for different tasks
        main_llm = LLMBackend(model_name="llama3:8b")
        curator_llm = LLMBackend(model_name="llama3:8b")
        reflector_llm = LLMBackend(model_name="llama3:8b") # This might need a stronger model in the future

        char_manager = CharacterManager(PERSONALITY_FILE)
        memory_manager = EpisodicMemoryManager(MEMORY_FILE)
        curator = Curator(curator_llm)
        reflector = Reflector(reflector_llm)
        guardrail = Guardrail()
    except FileNotFoundError as e:
        print(f" Critical Error: {e}. Bot cannot start.")
        return
    except Exception as e:
        print(f" An unexpected error occurred during initialization: {e}")
        return

    print("\n--- Potato Bot is Ready ---")
    print("Type 'quit' or 'exit' to end the chat.")
    
    turn_number = 0
    try:
        while True:
            turn_number += 1
            print(f"\n--- Turn {turn_number}: Waiting for message from Sota... ---")
            
            user_input = sota_client.receive_message_buffer()

            if user_input is None:
                print("Connection to Sota lost. Attempting to reconnect...")
                time.sleep(5)
                if not sota_client.connect():
                    print("Could not re-establish connection. Shutting down.")
                    break
                else:
                    print("Reconnected successfully.")
                    continue

            # Clean up the received message
            user_input = user_input.replace("\n", "").replace("result:", "").strip()
            print(f"Sota/User: {user_input}")

            if user_input.lower() in ["quit", "exit"]:
                print("Potato: Goodbye!")
                break

            # --- Main Response Generation ---
            # 1. Create a query embedding from the user's input
            query_embedding = MODELS.embedding_model.encode(user_input).tolist()

            # 2. Search for relevant memories (RAG)
            relevant_memories = memory_manager.search_memories(query_embedding, top_k=3)

            # 3. Construct the prompt for the main LLM
            system_prompt = char_manager.get_full_persona_text()
            
            user_prompt = "これが現在の会話です。\n"
            # We can add more chat history here if needed
            user_prompt += f"ユーザー: {user_input}\n"
            
            if relevant_memories:
                user_prompt += "\n過去の会話からの関連する記憶は次の通りです:\n"
                for mem in relevant_memories:
                    user_prompt += f"- {mem.curated_memory}\n"
            
            user_prompt += f"\nさて、{char_manager.persona.character.name}として、あなたのJSON応答は何ですか？"

            # 4. Call the main LLM
            bot_response_json = main_llm.call(system_prompt, user_prompt)

            if "error" in bot_response_json or "response_message" not in bot_response_json:
                bot_message = "ええと。。。なんて言ったらいいのかわからない。"
            else:
                bot_message = bot_response_json.get("response_message", "...")

            # 5. Check the response against the guardrail
            is_safe, final_message = guardrail.check(bot_message)
            
            print(f"Potato: {final_message}")

            # --- Send response back to Sota ---
            say_command = f"sota;say:{final_message}:rate=10:pitch=13:intonation=13"
            sota_client.send_command(say_command)
            
            end_turn_command = "sota;addAction:requestMessage:sh;end_gpt"
            sota_client.send_command(end_turn_command)
            print("--- Sent response to Sota. ---")


            # --- Post-Response Processing ---
            # 6. Curate a new memory from this turn
            current_turn = [
                ConversationTurn(speaker="User", message=user_input),
                ConversationTurn(speaker="Potato", message=final_message)
            ]
            new_memory_entry = curator.curate_memory_entry(current_turn, turn_number)
            if new_memory_entry:
                memory_manager.add_memory(new_memory_entry)

            # 7. Check if it's time to reflect
            if turn_number % REFLECTION_INTERVAL == 0:
                recent_memories = memory_manager.get_recent_memories(REFLECTION_INTERVAL)
                proposal = reflector.reflect_and_propose_change(char_manager.persona, recent_memories)
                
                if proposal:
                    # Update the in-memory persona dictionary
                    current_persona_dict = char_manager.persona.dict()
                    belief_to_update = proposal['belief_to_update']
                    new_belief = proposal['new_belief']

                    # Find and replace the belief
                    for i, belief in enumerate(current_persona_dict['character']['core_beliefs']):
                        if belief == belief_to_update:
                            current_persona_dict['character']['core_beliefs'][i] = new_belief
                            break
                    
                    # Save the updated persona back to the file
                    char_manager.update_and_save_persona(current_persona_dict)
    
    except KeyboardInterrupt:
        print("\n--- User interrupted. Shutting down. ---")
    finally:
        print("--- Closing connection to Sota. ---")
        sota_client.close()


if __name__ == "__main__":
    main()
