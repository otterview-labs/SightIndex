import uuid
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from app.schemas.common import SearchFilters
from app.schemas.media import SearchResultItem


class SemanticSearchRequest(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

    query: str = Field(min_length=1, max_length=2048)
    top_k: int = Field(default=20, ge=1, le=100)
    filters: SearchFilters = Field(default_factory=SearchFilters)


class SemanticSearchItem(SearchResultItem):
    match_type: Literal["semantic_candidate"] = "semantic_candidate"
    duplicate_crop_ids: list[uuid.UUID] = Field(default_factory=list)


class SemanticSearchResponse(BaseModel):
    mode: Literal["semantic"] = "semantic"
    items: list[SemanticSearchItem]
    model: str
    min_score: float
    notice: str
    scope_crops: int
    indexed_scope_crops: int
    candidates_examined: int = 0
    shortlist_limited: bool = False


class SemanticSearchStatus(BaseModel):
    enabled: bool
    configured: bool
    model: str
    min_score: float
    total_crops: int
    indexed_crops: int
    labeled_crops: int
    attributes_enabled: bool
    auto_index_on_ingest: bool
