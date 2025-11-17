import json
import os

def generate_conversation_log(memory_file_path: str, output_file_path: str):
    """
    Reads an episodic memory JSON file and writes a clean, human-readable
    conversation log to a text file.

    Args:
        memory_file_path: Path to the episodic_memory.json file.
        output_file_path: Path to the output .txt file to be created.
    """
    try:
        with open(memory_file_path, 'r', encoding='utf-8') as f:
            memories = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"Error reading memory file: {e}")
        return

    # Sort memories by turn number to ensure chronological order
    memories.sort(key=lambda x: x.get('turn_number', 0))

    with open(output_file_path, 'w', encoding='utf-8') as f:
        f.write("--- Conversation Log ---\n\n")
        
        for entry in memories:
            turn_number = entry.get('turn_number', 'N/A')
            f.write(f"--- Turn {turn_number} ---\n")
            
            conversation = entry.get('source_conversation', [])
            for turn in conversation:
                speaker = turn.get('speaker', 'Unknown')
                message = turn.get('message', '')
                f.write(f"{speaker}: {message}\n")
            
            # Also write the curated memory for context
            curated_memory = entry.get('curated_memory', 'No curated memory.')
            f.write(f"\n[Memory Summary]: {curated_memory}\n")
            f.write("="*20 + "\n\n")
            
    print(f"Successfully generated conversation log at: {output_file_path}")

if __name__ == "__main__":
    # --- Configuration ---
    # Get the absolute path of the directory where the script is located
    _script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Build the full path to the memory file relative to the script's location
    MEMORY_FILE = os.path.join(_script_dir, 'data', 'episodic_memory.json')
    
    # Set the output file to be in the same directory as the script
    OUTPUT_FILE = os.path.join(_script_dir, 'conversation_log.txt')
    
    generate_conversation_log(MEMORY_FILE, OUTPUT_FILE)
