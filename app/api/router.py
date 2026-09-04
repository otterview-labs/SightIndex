from fastapi import APIRouter

from app.api import (
    attributes,
    chat,
    embeddings,
    face,
    media,
    persons,
    reid,
    search,
    statistics,
    vlm,
)

api_router = APIRouter(prefix="/api")
api_router.include_router(persons.router)
api_router.include_router(media.router)
api_router.include_router(face.router)
api_router.include_router(reid.router)
api_router.include_router(attributes.router)
api_router.include_router(embeddings.router)
api_router.include_router(vlm.router)
api_router.include_router(search.router)
api_router.include_router(statistics.router)
api_router.include_router(chat.router)

openai_compatible_router = APIRouter()
openai_compatible_router.include_router(embeddings.openai_router)
