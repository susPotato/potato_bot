import uuid
from typing import List, Any
from pydantic import BaseModel, Field

def generate_uuid():
    """Generates a new UUID."""
    return str(uuid.uuid4())

class Memory(BaseModel):
    """
    A structured representation of a single memory, including its content,
    embedding vector, and a unique identifier.
    """
    id: str = Field(default_factory=generate_uuid)
    content: str
    embedding: List[float] = []

class MemoryFragment(BaseModel):
    """
    Represents a potential memory to be evaluated and possibly added to the store.
    It includes the content and the conversational turn it was derived from.
    """
    content: str
    turn_id: int # To link back to the conversation history

class CurationResult(BaseModel):
    """
    Defines the actions to be taken on the memory store after curation.
    """
    memories_to_add: List[Memory] = []
    ids_to_remove: List[str] = []
