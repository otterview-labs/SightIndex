from typing import Annotated

from fastapi import APIRouter, File, HTTPException, Query, UploadFile

from app.api.deps import AppSettings, DBSession
from app.schemas.persons import (
    FaceDiagnosticResponse,
    FaceLibraryRebuildResponse,
    FaceRecognitionRebuildResponse,
    FaceRecognitionResponse,
    FaceSearchResponse,
)
from app.services.faces import FaceRecognitionService

router = APIRouter(prefix="/face", tags=["face"])


UploadImage = Annotated[UploadFile, File(...)]


@router.post("/recognize")
def recognize_face(
    file: UploadImage,
    db: DBSession,
    settings: AppSettings,
    top_k: int = Query(default=5, ge=1, le=50),
    threshold: float | None = Query(default=None, ge=0.0, le=1.0),
) -> FaceRecognitionResponse:
    try:
        return FaceRecognitionService(db, settings).recognize_upload(
            file=file,
            top_k=top_k,
            threshold=threshold,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/search")
def search_face(
    file: UploadImage,
    db: DBSession,
    settings: AppSettings,
    top_k: int = Query(default=10, ge=1, le=100),
    min_similarity: float | None = Query(default=None, ge=0.0, le=1.0),
) -> FaceSearchResponse:
    try:
        return FaceRecognitionService(db, settings).search_upload(
            file=file,
            top_k=top_k,
            min_similarity=min_similarity,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/cluster-unknown")
def cluster_unknown_faces() -> dict[str, str]:
    return {"status": "stub", "message": "Unknown clustering is planned for phase B."}


@router.get("/diagnostics/recent", response_model=FaceDiagnosticResponse)
def diagnose_recent_face_crops(
    db: DBSession,
    settings: AppSettings,
    limit: int = Query(default=20, ge=1, le=100),
) -> FaceDiagnosticResponse:
    service = FaceRecognitionService(db, settings)
    return FaceDiagnosticResponse(
        threshold=settings.face_match_threshold,
        items=service.diagnose_recent_crops(limit=limit),
    )


@router.post("/index/rebuild", response_model=FaceRecognitionRebuildResponse)
def rebuild_face_recognition_index(
    db: DBSession,
    settings: AppSettings,
    limit: int = Query(default=500, ge=1, le=10000),
    force: bool = Query(default=False),
) -> FaceRecognitionRebuildResponse:
    result = FaceRecognitionService(db, settings).rebuild_crop_recognition(
        limit=limit,
        force=force,
    )
    return FaceRecognitionRebuildResponse(**result)


@router.post("/library/rebuild", response_model=FaceLibraryRebuildResponse)
def rebuild_face_library(
    db: DBSession,
    settings: AppSettings,
    limit: int = Query(default=500, ge=1, le=10000),
    allow_fallback: bool = Query(default=False),
) -> FaceLibraryRebuildResponse:
    result = FaceRecognitionService(db, settings).rebuild_library_embeddings(
        limit=limit,
        allow_fallback=allow_fallback,
    )
    return FaceLibraryRebuildResponse(**result)
