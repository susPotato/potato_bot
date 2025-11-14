import uuid
from typing import List, Dict, Any
from pydantic import BaseModel, Field
from datetime import datetime

# --- Layer 1: Core Persona Schemas ---

class CharacterDetails(BaseModel):
    name: str
    persona: str
    core_beliefs: List[str]
    speech_patterns: 'SpeechPatterns'

class SpeechPatterns(BaseModel):
    use_short_sentences: bool
    tone: str
    common_phrases: List[str]

class SampleDialog(BaseModel):
    speaker: str
    message: str

class CorePersona(BaseModel):
    character: CharacterDetails
    knowledge_base: Dict[str, str]
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
