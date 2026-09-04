import uuid
from datetime import datetime

from pydantic import BaseModel, Field


class ReidMatchItem(BaseModel):
    crop_id: uuid.UUID
    score: float
    image_id: uuid.UUID | None = None
    crop_url: str | None = None
    image_url: str | None = None
    captured_at: datetime | None = None
    camera_id: uuid.UUID | None = None
    camera_name: str | None = None
    location_id: uuid.UUID | None = None
    location_name: str | None = None
    person_id: uuid.UUID | None = None
    person_name: str | None = None
    # Where this body ranked in its own camera's crowd by height, and how closely that agrees
    # with the query. Reported so a cross-camera match can be read rather than just trusted.
    stature_percentile: int | None = None
    stature_agreement: float | None = None
    attribute_agreement: float | None = None
    attribute_matches: list[str] = Field(default_factory=list)
    attribute_conflicts: list[str] = Field(default_factory=list)
    attribute_comparable_count: int = 0
    attribute_match_count: int = 0
    attribute_conflict_count: int = 0
    attribute_evidence_weight: float | None = None
    attribute_conflict_weight: float | None = None
    # Face evidence is optional: None means one side had no reliable face. True gets the highest
    # rank; False is shown as contrary evidence but does not delete a body match by itself.
    face_similarity: float | None = None
    face_match: bool | None = None
    face_query_quality: float | None = None
    face_candidate_quality: float | None = None
    face_reliability: float | None = None
    # Backend-owned explanation of the final ordering.  This is deliberately a score, not a
    # probability: calibration has not yet shown that 0.7 means a 70% identity likelihood.
    fusion_score: float | None = None
    evidence_level: str | None = None
    decision_reason: str | None = None
    # A match stands for a visit, not a frame: how many frames it merged and when it ran.
    frame_count: int = 1
    first_seen: datetime | None = None
    last_seen: datetime | None = None


class ReidSearchResponse(BaseModel):
    items: list[ReidMatchItem]
    model: str
    min_score: float
    collapse_window_seconds: float = 0.0
    query_mode: str = "single_frame"
    query_frame_count: int = 1


class ReidCameraLink(BaseModel):
    """The likeliest appearance of this person at one other camera.

    Not a match: no threshold decides whether it is returned. It answers "if they went there,
    which one were they" and leaves the judgement to whoever is looking, which is the only
    honest thing to do when a real crossing scores 0.43-0.48 and chance reaches 0.44.
    """

    camera_id: uuid.UUID | None = None
    camera_name: str | None = None
    location_name: str | None = None
    crop_id: uuid.UUID
    crop_url: str | None = None
    # How closely this candidate's height rank matches the query's, 1 identical and 0 fifty
    # percentile points apart. None when either side's height could not be measured.
    stature_agreement: float | None = None
    attribute_agreement: float | None = None
    attribute_matches: list[str] = Field(default_factory=list)
    attribute_conflicts: list[str] = Field(default_factory=list)
    attribute_comparable_count: int = 0
    attribute_match_count: int = 0
    attribute_conflict_count: int = 0
    attribute_evidence_weight: float | None = None
    attribute_conflict_weight: float | None = None
    face_similarity: float | None = None
    face_match: bool | None = None
    face_query_quality: float | None = None
    face_candidate_quality: float | None = None
    face_reliability: float | None = None
    fusion_score: float | None = None
    evidence_level: str | None = None
    decision_reason: str | None = None
    score: float
    captured_at: datetime | None = None
    # True when the score is out of reach of coincidence, measured against pairs that cannot be
    # the same person. False means "this is the best there is", not "this is them".
    beats_chance: bool


class ReidLinkResponse(BaseModel):
    crop_id: uuid.UUID
    camera_id: uuid.UUID | None = None
    camera_name: str | None = None
    captured_at: datetime | None = None
    links: list[ReidCameraLink]
    chance_ceiling: float
    query_mode: str = "single_frame"
    query_frame_count: int = 1


class ReidStatusResponse(BaseModel):
    # `enabled` keeps its original meaning: the feature is configured. `ready` is the honest
    # bit: the ReID service answered a live health probe with a matching model identity AND
    # Milvus answered a bounded connect. A fresh process against an unreachable Milvus is
    # not ready, whatever the configuration says.
    enabled: bool
    ready: bool = False
    reid_service_ok: bool = False
    milvus_configured: bool = False
    milvus_ok: bool = False
    milvus_in_cooldown: bool = False
    last_error: str | None = None
    service_url: str | None = None
    model: str
    checkpoint_revision: str = ""
    embedding_dim: int
    preprocess_version: str = ""
    milvus_namespace: str = ""
    index_fingerprint: str = ""
    indexed_crops: int
    pending_crops: int
    min_score: float
    attribute_filter_enabled: bool = False
    attribute_min_confidence: float = 0.0
    attribute_hard_conflicts: int = 0
    attribute_hard_confidence: float = 0.0
    attribute_full_weight: float = 0.0
    attribute_hard_filter_window_hours: float = 0.0
    face_priority_enabled: bool = False
    face_priority_ready: bool = False
    face_priority_error: str | None = None
    face_provider: str = ""
    face_model: str = ""
    face_device: str = ""
    face_candidate_limit: int = 0
    face_min_quality: float = 0.0
    face_strong_reliability: float = 0.0


class ReidRebuildResponse(BaseModel):
    requested: int
    seen: int
    indexed: int
    # seen == indexed + skipped + failed + unprocessed: files gone, embed/index failures,
    # and crops never reached because the batch loop aborted, counted separately.
    skipped: int
    failed: int = 0
    unprocessed: int = 0
    errors: list[str]
