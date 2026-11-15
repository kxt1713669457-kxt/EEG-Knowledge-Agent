from app.shared.clients.mongo_history_utils import (
    clear_history,
    get_recent_messages,
    save_chat_message,
    update_message_item_names,
)


class HistoryRepository:
    def list_recent(self, session_id: str, limit: int = 10) -> list[dict]:
        return get_recent_messages(session_id, limit=limit)

    def save_message(
        self,
        *,
        session_id: str,
        role: str,
        text: str,
        rewritten_query: str = "",
        item_names: list[str] | None = None,
        image_urls: list[str] | None = None,
        eeg_attachments: list[dict] | None = None,
        message_id: str | None = None,
        citations: list[dict] | None = None,
        citation_status: dict | None = None,
        evidence_sources: list[dict] | None = None,
        answer_audit: dict | None = None,
        task_id: str = "",
        run_id: str | None = None,
        input_revision: int = 1,
        message_kind: str = "",
    ) -> str:
        return save_chat_message(
            session_id=session_id,
            role=role,
            text=text,
            rewritten_query=rewritten_query,
            item_names=item_names,
            image_urls=image_urls,
            eeg_attachments=eeg_attachments,
            message_id=message_id,
            citations=citations,
            citation_status=citation_status,
            evidence_sources=evidence_sources,
            answer_audit=answer_audit,
            task_id=task_id,
            run_id=run_id,
            input_revision=input_revision,
            message_kind=message_kind,
        )

    def clear_session(self, session_id: str) -> int:
        return clear_history(session_id)



history_repository = HistoryRepository()
