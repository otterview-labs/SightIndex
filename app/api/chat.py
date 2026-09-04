from fastapi import APIRouter

from app.api.deps import AppSettings, DBSession
from app.schemas.chat import ChatRequest, ChatResponse
from app.services.chat import ChatService

router = APIRouter(prefix="/chat", tags=["chat"])


@router.post("", response_model=ChatResponse)
def chat(payload: ChatRequest, db: DBSession, settings: AppSettings) -> ChatResponse:
    return ChatService(db, settings).handle(payload.message, payload.session_id, payload.context)
