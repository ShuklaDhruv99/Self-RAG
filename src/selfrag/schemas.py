from pydantic import BaseModel, Field
from typing import Literal

class RetrieveDecision(BaseModel):
    should_retrieve: bool = Field(
        ...,
        description="True if external documents are needed to answer reliably, else False."
    )

class RelevanceGrade(BaseModel):
    reason: str = Field(..., description="Short reason in 1 line.")
    is_relevant: bool = Field(
        ...,
        description="True ONLY if the document contains info that helps answer the question."
    )

class SupportGrade(BaseModel):
    evidence: list[str]
    is_sup: Literal["fully_supported", "partially_supported", "no_support"]

class UsefulnessGrade(BaseModel):
    reason: str = Field(..., description="Short reason in 1 line.")
    is_use: Literal["useful", "not_useful"]
