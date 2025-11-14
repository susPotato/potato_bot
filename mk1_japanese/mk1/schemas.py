import uuid
from typing import List, Dict, Any
from pydantic import BaseModel, Field
from datetime import datetime

# --- Layer 1: Core Persona Schemas ---

class CharacterDetails(BaseModel):
    name: str
    persona: str
    internal_conflict: str
    core_beliefs: List[str]
    speech_patterns: 'SpeechPatterns'

class SpeechPatterns(BaseModel):
    use_short_sentences: bool
    tone: str
    show_dont_tell: str

class InteractionRules(BaseModel):
    your_hidden_goal: str
    on_receiving_simple_platitudes: str
    on_receiving_genuine_questions: str
    on_receiving_insults: str
    addressing_the_user: str

class SampleDialog(BaseModel):
    speaker: str
    message: str

class CorePersona(BaseModel):
    character: CharacterDetails
    knowledge_base: Dict[str, str]
    interaction_rules: InteractionRules
    sample_dialog: List[SampleDialog]


# --- Layer 2: Episodic Memory Schemas ---

class ConversationTurn(BaseModel):
    speaker: str
    message: str

class EpisodicMemoryEntry(BaseModel):
    id: str = Field(default_factory=lambda: f"mem_{uuid.uuid4()}")
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    turn_number: int
    source_conversation: List[ConversationTurn]
    curated_memory: str
    emotional_valence: str
    embedding: List[float] = []

class CurationResult(BaseModel):
    """
    Defines the output of the Curator's analysis for a single turn.
    """
    memory_entry: EpisodicMemoryEntry
