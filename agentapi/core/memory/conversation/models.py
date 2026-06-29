from __future__ import annotations

from dataclasses import dataclass,field
from datetime import datetime
from typing import Any
from uuid import uuid4

@dataclass(slots=True)
class Conversation:
    id:str =  field(default_factory=lambda:str(uuid4()))
    created_at:datetime = field(default_factory=datetime.now)
    updated_at:datetime = field(default_factory=datetime.now)
    
class ConversationRecord:
    id:str = field(default_factory=lambda:str(uuid4()))
    conversation_id:str = ""
    role:str = ""
    content:str = ""
    metadata:dict[str,Any] = field(default_factory=dict)
    created_at:datetime = field(default_factory=datetime.utcnow)
    
    