from typing import List, Literal
from pydantic import BaseModel, Field

# Defining message schema
class Message(BaseModel):
    role: str
    content: str

# List of messages
class ChatRequest(BaseModel):
    messages: List[Message]

# recommendation item schema
class RecItem(BaseModel):
    name: str
    url: str
    test_type: str

# Chat response schema
class ChatResponse(BaseModel):
    reply: str
    recommendations: List[RecItem] = []
    conversation_end: bool

# Internal routing schema for agent
class IntentRouting(BaseModel):
    intent: Literal["clarify", "recommend", "refine", "compare", "refuse"] = Field(description="The primary conversational intent of the user step.")
    reasoning: str = Field(description="Brief explanation of the routing choice.")
    reply_content: str = Field(description="The response message to be sent to the user.")
    keywords: List[str] = Field(default=[], description="Search phrases used to pull matching items from the index.")
    show_recs: bool = Field(description="True if shortlists should be generated this turn, False otherwise.")
    end_conv: bool = Field(description="True if the interaction is successfully completed.")