from __future__ import annotations

from typing import List, Literal, Optional

from pydantic import BaseModel, Field


class TokenResponse(BaseModel):
    access_token: str
    token_type: Literal["bearer"]


class ChatRequest(BaseModel):
    question: str = Field(min_length=1, description="User question to answer.")
    top_k: int = Field(default=5, ge=1, le=50)


class ChatResponse(BaseModel):
    answer: str
    sources: List[str] = Field(default_factory=list)
