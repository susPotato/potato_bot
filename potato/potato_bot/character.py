import json
import os
import pickle
import ollama
import datetime
import numpy as np
from typing import List, Tuple

def cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
    """
    Calculates the cosine similarity between two vectors.
    """
    vec1 = np.array(vec1)
    vec2 = np.array(vec2)
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

class Character:
    def __init__(self, character_file: str, chat_history_limit=10) -> None:
        with open(character_file) as f:
            self.character_info = json.loads(f.read())
        self.chat_history_limit = chat_history_limit
        self.name = self.character_info["character"]["name"]
        self.savefile = character_file

        self.kb_embed_file = f"{self.name}.kb"
        self.memory_embed_file = f"{self.name}.memories"

        # Knowledge base embeddings
        if os.path.exists(self.kb_embed_file):
            print("loading knowledge base embeddings from file...")
            self.kb_embed_pairs = self.load_embeddings(self.kb_embed_file)
        else:
            print("embedding knowledge base...")
            kb_items = list(self.character_info["knowledge_base"].items())
            kb_embeddings = [self.get_embedding(f"{key}: {definition}") for key, definition in kb_items]
            self.kb_embed_pairs = list(zip(kb_items, kb_embeddings))
            self.save_embeddings(self.kb_embed_file, self.kb_embed_pairs)

        # Memory embeddings
        if os.path.exists(self.memory_embed_file):
            print("loading memory embeddings from file...")
            self.memory_embeddings = self.load_embeddings(self.memory_embed_file)
        else:
            print("embedding memories...")
            memory_contents = [mem.split('|', 1)[1] for mem in self.character_info["memories"]]
            self.memory_embeddings = [self.get_embedding(content) for content in memory_contents]
            self.save_embeddings(self.memory_embed_file, self.memory_embeddings)

    def save_embeddings(self, filename, data):
        with open(filename, 'wb') as f:
            pickle.dump(data, f)

    def load_embeddings(self, filename):
        with open(filename, 'rb') as f:
            return pickle.load(f)

    def get_embedding(self, text: str):
        """
        Generates an embedding for the given text using a local Ollama model.
        """
        response = ollama.embeddings(
            model='nomic-embed-text',
            prompt=text
        )
        return response["embedding"]

    def normalize_scores(self, scores: List[float]) -> List[float]:
        """
        Normalize the scores to a range between 0 and 1.
        """
        min_score = min(scores)
        max_score = max(scores)
        if max_score == min_score:  # Prevent division by zero
            return [0.5] * len(scores)
        return [(s - min_score) / (max_score - min_score) for s in scores]

    def rank_memories_by_similarity_and_recency(
        self,
        query: str,
        memories: List[str],
        alpha: float = 0.5
    ) -> List[Tuple[str, float]]:
        query_embedding = self.get_embedding(query)

        # Separate timestamps and memory content
        timestamps = [datetime.datetime.strptime(mem.split("|", 1)[0], "%d-%m-%y %H:%M:%S") for mem in memories]
        similarity_scores = [cosine_similarity(query_embedding, embedding) for embedding in self.memory_embeddings]

        # Normalize similarity scores
        normalized_similarity_scores = self.normalize_scores(similarity_scores)

        # Compute recency scores (normalize so that newer memories have higher scores)
        now = datetime.datetime.now()
        time_deltas = [(now - ts).total_seconds() for ts in timestamps]  # Time differences in seconds
        recency_scores = self.normalize_scores([-delta for delta in time_deltas])  # Negative deltas to prefer recent memories

        # Combine similarity and recency scores using alpha
        combined_scores = [
            alpha * sim + (1 - alpha) * rec
            for sim, rec in zip(normalized_similarity_scores, recency_scores)
        ]

        # Combine scores with memory content and sort
        ranked_memories = sorted(
            zip(memories, combined_scores),
            key=lambda x: x[1],
            reverse=True
        )

        return ranked_memories

    def find_most_relevant_knowledge_base(
        self,
        query: str,
        knowledge_base: dict,
        top_n: int = 3
    ) -> List[Tuple[str, str, float]]:
        """
        Finds the most relevant knowledge base entries for a given query.

        Returns a list of tuples (key, definition, score), sorted by similarity.
        """
        query_embedding = self.get_embedding(query)

        # Calculate similarity for each key-definition
        similarities = [
            (key, definition, cosine_similarity(query_embedding, embedding))
            for (key, definition), embedding in self.kb_embed_pairs
        ]

        # Sort by similarity and return top N
        similarities = sorted(similarities, key=lambda x: x[2], reverse=True)
        return similarities[:top_n]

    def render_prompt(self, messages):
        info_copy = dict()
        info_copy["character"] = self.character_info["character"]
        info_copy["sample_dialog"] = self.character_info["sample_dialog"]
        info_copy["known_people"] = self.character_info["known_people"]
        info_copy["important_memories"] = self.character_info["important_memories"]

        query = ""
        messages = messages[-self.chat_history_limit:]
        # last_message = messages.pop()
        for message in messages:
            if message["speaker"] != self.name:
                query += f"{message["speaker"]}: {message["message"]} "

        memories = list()
        definitions = dict()
        if len(query) > 0:
            memories_ranked = self.rank_memories_by_similarity_and_recency(query, self.character_info["memories"], alpha=0.7)[:5]
            for mem, score in memories_ranked:
                if score > 0.4:
                    memories.append(mem.split('|', 1)[1])

            definitions_ranked = self.find_most_relevant_knowledge_base(query, self.character_info["knowledge_base"])
            for key, definition, similarity in definitions_ranked:
                if similarity > 0.4:
                    definitions[key] = definition

        system = f"Act as {self.name}. Here's all the information about {self.name}:\n{info_copy}\n"
        if len(memories) > 0:
            system += f"Potentially relevant memories:\n{memories}\n"
        if len(definitions) > 0:
            # messages[-1]["potentially_relevant_definitions"] = definitions
            system += f"Potentially relevant definitions:\n{definitions}\n"
        system += f"You are in a discord group chat. This is the conversation you are currently participating in:\n{messages[-self.chat_history_limit:]}\n"
        prompt = f"Respond in the following json format as {self.name}:"
        prompt += '''
{
    "thoughts": str,
    "should_respond": bool,
    "response_message": string,
    "known_people_updates": dict,
    "knowledge_base_updates": dict,
    "new_memories": list
}
where "thoughts" is your thoughts the last message in the conversation. Only think about the last message in the conversation. Based on it, brainstorm what you should respond with if you choose to respond and whether to respond or not. Brainstorm well whether to respond or not based on the context of the chat. "should_respond" is whether you should respond to the last message in the current conversation or not. Only return true if there is a direct mention of you, or if you feel interested to jump in on the topic being discussed. "response_message" is the message that you want to respond with if you choose to respond, based on your thoughts, and the conversation context. "known_people_updates" is the updates you want to make to the fields of what you know about your known people in your character description, including new information about their character, thier relation to you, etc (use the username as the key, and keep this very short, don't add information that is irrevant or already present in memories). "new_memories" is a list of new memories (strings) to add if there's anything new and important worth remembering from the last message in the chat. ONLY ADD VERY IMPORTANT THINGS TO YOUR MEMORY ONLY IF NECESSARY. "knowledge_base_updates" is updates to information, or new information and terms, nouns, websites, etc. that you learned about FROM THE CHAT, where the key is the term, and the value is the definition/explanation. Only add new things you learned from the chat ONLY IF THEY ARE IMPORTANT TO YOU.
'''
        prompt += f"Rember to act as {self.name} and follow your character description, but don't copy dialogs from the sample directly (be creative in your response, but be short since this is a chat). Don't repeat yourself."
        # prompt += f"\nNew message to respond to:\n{last_message}"
        return system, prompt

    def update_info(self, info_json):
        if "known_people_updates" in info_json:
            for person in info_json["known_people_updates"]:
                if person not in self.character_info["known_people"]:
                    self.character_info["known_people"][person] = info_json["known_people_updates"][person]
                else:
                    for key in info_json["known_people_updates"][person]:
                        self.character_info["known_people"][person][key] = info_json["known_people_updates"][person][key]
        if "new_memories" in info_json:
            for mem in info_json["new_memories"]:
                if mem == self.character_info["memories"][-1].split('|', 1)[1]:
                    continue
                self.character_info["memories"].append(datetime.datetime.now().strftime('%d-%m-%y %H:%M:%S') + '|' + mem)
                self.memory_embeddings.append(self.get_embedding(mem))
                self.save_embeddings(self.memory_embed_file, self.memory_embeddings)
        if "knowledge_base_updates" in info_json:
            for item in info_json["knowledge_base_updates"]:
                self.character_info["knowledge_base"][item] = info_json["knowledge_base_updates"][item]
                self.kb_embed_pairs.append(((item, info_json["knowledge_base_updates"][item]), self.get_embedding(f"{item}: {info_json["knowledge_base_updates"][item]}")))
                self.save_embeddings(self.kb_embed_file, self.kb_embed_pairs)

        with open(self.savefile, "w+") as f:
            f.write(json.dumps(self.character_info, indent=4))
