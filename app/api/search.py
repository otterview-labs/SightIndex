import tempfile
import uuid
from datetime import datetime
from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, File, HTTPException, Query, UploadFile

from app.api.deps import AppSettings, DBSession
from app.schemas.common import SearchFilters
from app.schemas.media import (
    ImageSearchRequest,
    IndexRebuildResponse,
    ObservationIndexItem,
    ObservationIndexResponse,
    SearchResponse,
    VisualSearchRequest,
)
from app.schemas.semantic_search import (
    SemanticSearchRequest,
    SemanticSearchResponse,
    SemanticSearchStatus,
)
from app.services.observation_index import ObservationIndexService
from app.services.search import VisualSearchService
from app.services.semantic_search import SemanticSearchService, SemanticSearchUnavailable
from app.services.vector_index import VectorIndexError, VectorIndexingService

router = APIRouter(prefix="/search", tags=["search"])


@router.get("/semantic/status", response_model=SemanticSearchStatus)
def semantic_search_status(db: DBSession, settings: AppSettings) -> SemanticSearchStatus:
    return SemanticSearchService(db, settings).status()


@router.post("/semantic/person-crops", response_model=SemanticSearchResponse)
def semantic_search_person_crops(
    payload: SemanticSearchRequest, db: DBSession, settings: AppSettings
) -> SemanticSearchResponse:
    try:
        return SemanticSearchService(db, settings).search(payload)
    except SemanticSearchUnavailable as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/images", response_model=SearchResponse)
def search_images(
    payload: VisualSearchRequest,
    db: DBSession,
    settings: AppSettings,
) -> SearchResponse:
    payload.target = "image"
    try:
        return VisualSearchService(db, settings).search(payload)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/person-crops", response_model=SearchResponse)
def search_person_crops(
    payload: VisualSearchRequest,
    db: DBSession,
    settings: AppSettings,
) -> SearchResponse:
    payload.target = "person_crop"
    try:
        return VisualSearchService(db, settings).search(payload)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/index/rebuild", response_model=IndexRebuildResponse)
def rebuild_search_index(
    db: DBSession,
    settings: AppSettings,
    target: str = Query(default="person_crop", pattern="^(image|person_crop)$"),
    limit: int = Query(default=500, ge=1, le=10000),
) -> IndexRebuildResponse:
    try:
        result = VectorIndexingService(db, settings).rebuild(target=target, limit=limit)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return IndexRebuildResponse(**result)


@router.post("/observations/rebuild")
def rebuild_observation_index(
    db: DBSession,
    settings: AppSettings,
    limit: int = Query(default=1000, ge=1, le=100000),
) -> dict[str, object]:
    return ObservationIndexService(db, settings).rebuild(limit=limit)


@router.get("/observations", response_model=ObservationIndexResponse)
def list_observations(
    db: DBSession,
    settings: AppSettings,
    limit: Annotated[int, Query(ge=1, le=500)] = 100,
    offset: Annotated[int, Query(ge=0)] = 0,
    query: str | None = None,
    person_id: uuid.UUID | None = None,
    camera_id: uuid.UUID | None = None,
    location_id: uuid.UUID | None = None,
    start_time: datetime | None = None,
    end_time: datetime | None = None,
    only_named: bool = False,
    only_face_vector: bool = False,
    only_vl_vector: bool = False,
    only_labeled: bool = False,
) -> ObservationIndexResponse:
    filters = SearchFilters(
        person_id=person_id,
        camera_id=camera_id,
        location_id=location_id,
        start_time=start_time,
        end_time=end_time,
    )
    try:
        rows, total = ObservationIndexService(db, settings).list_rows(
            limit=limit,
            offset=offset,
            query=query,
            filters=filters,
            only_named=only_named,
            only_face_vector=only_face_vector,
            only_vl_vector=only_vl_vector,
            only_labeled=only_labeled,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return ObservationIndexResponse(
        items=[ObservationIndexItem.model_validate(row, from_attributes=True) for row in rows],
        total=total,
        limit=limit,
        offset=offset,
    )


@router.post("/videos", response_model=SearchResponse)
def search_videos() -> SearchResponse:
    return SearchResponse(items=[])


@router.post("/by-image", response_model=SearchResponse)
def search_by_image(
    payload: ImageSearchRequest,
    db: DBSession,
    settings: AppSettings,
) -> SearchResponse:
    return VisualSearchService(db, settings).search_by_image(payload)


@router.post("/by-image/upload", response_model=SearchResponse)
def search_by_image_upload(
    db: DBSession,
    settings: AppSettings,
    file: Annotated[UploadFile, File(...)],
    top_k: int = Query(default=10, ge=1, le=100),
    min_score: float | None = Query(default=None, ge=0.0, le=1.0),
) -> SearchResponse:
    """Rank indexed images by similarity to an uploaded query image.

    Answers "where does this appear" for partial queries -- a torso crop, an object
    in hand -- which the ReID body-identity channel deliberately does not serve.
    Works over the visual-embedding vectors ingest already built for images.
    """

    service = VisualSearchService(db, settings)
    suffix = Path(file.filename or "query.jpg").suffix or ".jpg"
    with tempfile.TemporaryDirectory(prefix="sightindex-imgsearch-") as temp_dir:
        query_path = Path(temp_dir) / f"query{suffix}"
        query_path.write_bytes(file.file.read())
        try:
            return service.search_by_uploaded_image(query_path, top_k, min_score)
        except VectorIndexError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
