import uuid
from datetime import datetime
from typing import Annotated

from fastapi import APIRouter, HTTPException, Query

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
from app.services.observation_index import ObservationIndexService
from app.services.search import VisualSearchService
from app.services.vector_index import VectorIndexingService

router = APIRouter(prefix="/search", tags=["search"])


@router.post("/images", response_model=SearchResponse)
def search_images(
    payload: VisualSearchRequest,
    db: DBSession,
    settings: AppSettings,
) -> SearchResponse:
    payload.target = "image"
    return VisualSearchService(db, settings).search(payload)


@router.post("/person-crops", response_model=SearchResponse)
def search_person_crops(
    payload: VisualSearchRequest,
    db: DBSession,
    settings: AppSettings,
) -> SearchResponse:
    payload.target = "person_crop"
    return VisualSearchService(db, settings).search(payload)


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
