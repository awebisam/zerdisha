"""Pydantic models for structured LLM responses (Responses API schemas)."""

from typing import List, Literal
from pydantic import BaseModel, Field


class SeedItem(BaseModel):
    """Single exploration seed suggestion."""
    concept: str = Field(..., description="Seed concept name or theme")
    discovery_type: Literal[
        "connection_gap", "adjacent_domain", "deeper_layer", "cross_pollination"
    ] = Field(..., description="Type of discovery opportunity")
    rationale: str = Field(..., description="Why this seed is valuable")
    related_concepts: List[str] = Field(default_factory=list)
    suggested_questions: List[str] = Field(default_factory=list)
    priority: float = Field(default=0.5, ge=0.0, le=1.0)


class SeedResponse(BaseModel):
    """Structured response containing a list of seeds."""
    seeds: List[SeedItem] = Field(default_factory=list)


# Optional: Future models for MA analysis/finalization can be added here.
# Keeping file small/specific for now to minimize risk of breaking changes.
