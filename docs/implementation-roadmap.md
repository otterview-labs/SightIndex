# SightIndex Implementation Roadmap

## Phase A: Runnable API MVP

Status: started.

Deliverables:

- FastAPI app with route contracts from the solution document.
- PostgreSQL schema equivalent in SQLAlchemy.
- Local file storage under `data/`.
- RTSP/HTTP video stream registration and optional OpenCV frame capture.
- OpenCV HOG person-triggered frame retention, so empty frames do not flood storage.
- Placeholder visual search, face recognition, statistics, and chat services.
- API smoke tests.

Exit criteria:

- `pytest` passes.
- `uvicorn main:app --reload` starts.
- Users can create persons, register streams, capture frames, call chat, and see explicit stub
  responses for model-backed features.

## Phase B: Real Image Ingestion

Deliverables:

- Stream capture creates `images` rows with `source_type=stream_frame` only when a person is detected.
- YOLO person detector creates accurate `person_crops` rows and crop files from captured frames.
- Optional thumbnail generation.
- Recognition/counting event creation with dedup rules.

## Phase C: Vector Retrieval

Deliverables:

- Milvus collections for image and person-crop embeddings.
- Pluggable visual embedding runtime with SentenceTransformers CLIP baseline and Qwen3-VL adapter.
- OpenAI-compatible VLM caption indexing for deployments that expose chat/completions but not
  embeddings.
- `/api/search/person-crops` uses dense retrieval plus SQL filters.
- `/api/search/by-image` embeds an existing query image/crop and searches person crops.
- Face crops and enrolled face vectors use separate Milvus collections and can degrade to SQL when
  the vector service is unavailable.

## Phase D: Face Identity

Deliverables:

- InsightFace enrollment for `/api/persons/{id}/faces`, with optional Milvus mirroring for fast
  identity search and `/api/face/library/rebuild` backfill.
- `/api/face/recognize` identity search with threshold handling.
- Unknown cluster persistence.

## Phase E: Chat Tool Orchestration

Deliverables:

- Intent parser/tool planner.
- Approved tool registry.
- Tool trace persisted to `chat_messages`.
- Query expansion for Chinese/English visual descriptions.
