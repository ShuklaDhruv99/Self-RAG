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

SupportLabel = Literal["fully_supported", "partially_supported", "no_support"]

class SupportGrade(BaseModel):
    evidence: list[str] = Field(
        default_factory=list,
        description="Up to 3 short verbatim quotes from the context that support the answer. Empty if none."
    )
    is_sup: SupportLabel = Field(
        ...,
        description="fully_supported: every claim is stated in the context. "
                    "partially_supported: core facts are supported but the answer adds a claim or judgment the context does not state. "
                    "no_support: the key claims are not in the context."
    )

class UsefulnessGrade(BaseModel):
    reason: str = Field(..., description="Short reason in 1 line.")
    is_use: Literal["useful", "not_useful"]
