import uuid
from typing import Any

from sqlalchemy.orm import Session

from app.config.settings import Settings, get_settings
from app.models.chat import ChatMessage
from app.schemas.chat import ChatResponse
from app.services.chat_tools import ChatToolContext, default_chat_tools
from app.services.search import StructuredSearchService, VisualSearchService
from app.services.statistics import StatisticsService


class ChatService:
    def __init__(self, db: Session, settings: Settings | None = None) -> None:
        self.db = db
        self.settings = settings or get_settings()
        self.search = VisualSearchService(db, self.settings)
        self.structured_search = StructuredSearchService(db, self.settings)
        self.statistics = StatisticsService(db)
        self.tools = default_chat_tools()

    def handle(
        self,
        message: str,
        session_id: uuid.UUID | None = None,
        context: dict[str, Any] | None = None,
    ) -> ChatResponse:
        session_id = session_id or uuid.uuid4()
        self._save_message(session_id, "user", message)
        tool_context = ChatToolContext(
            db=self.db,
            settings=self.settings,
            search=self.search,
            structured_search=self.structured_search,
            statistics=self.statistics,
            request_context=context or {},
        )
        response = self._run_tool(message, tool_context)
        self._save_message(
            session_id,
            "assistant",
            response.answer,
            response.tool_name,
            response.tool_params,
            response.data,
        )
        self.db.commit()
        return response

    def _run_tool(self, message: str, context: ChatToolContext) -> ChatResponse:
        for tool in self.tools:
            if tool.can_handle(message, context):
                return tool.run(message, context)
        raise RuntimeError("No chat tool is registered")

    def _save_message(
        self,
        session_id: uuid.UUID,
        role: str,
        content: str,
        tool_name: str | None = None,
        tool_params: dict[str, Any] | None = None,
        tool_result: dict[str, Any] | None = None,
    ) -> None:
        self.db.add(
            ChatMessage(
                session_id=session_id,
                role=role,
                content=content,
                tool_name=tool_name,
                tool_params=tool_params,
                tool_result=tool_result,
            )
        )
