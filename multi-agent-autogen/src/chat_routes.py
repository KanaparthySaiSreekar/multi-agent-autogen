import json
import logging
import uuid

from fastapi import APIRouter, File, HTTPException, UploadFile
from fastapi.responses import StreamingResponse

from src.chat_models import ChatRequest
from src.database import get_db
from src.orchestrator import analyze_document_streaming

logger = logging.getLogger(__name__)
chat_router = APIRouter(prefix="/api/chat")


async def _ensure_conversation(conversation_id: str | None, first_message: str) -> str:
    """Return an existing conversation id or create a new one."""
    db = await get_db()
    try:
        if conversation_id:
            cursor = await db.execute(
                "SELECT id FROM conversations WHERE id = ?", (conversation_id,)
            )
            if await cursor.fetchone():
                await db.execute(
                    "UPDATE conversations SET updated_at = CURRENT_TIMESTAMP WHERE id = ?",
                    (conversation_id,),
                )
                await db.commit()
                return conversation_id

        cid = uuid.uuid4().hex
        name = first_message[:60].strip() or "New Chat"
        await db.execute(
            "INSERT INTO conversations (id, name) VALUES (?, ?)", (cid, name)
        )
        await db.commit()
        return cid
    finally:
        await db.close()


async def _save_message(
    conversation_id: str,
    role: str,
    content: str | None = None,
    component: str | None = None,
    payload: dict | None = None,
):
    db = await get_db()
    try:
        await db.execute(
            "INSERT INTO chat_messages (conversation_id, role, content, component, payload) VALUES (?, ?, ?, ?, ?)",
            (
                conversation_id,
                role,
                content,
                component,
                json.dumps(payload) if payload else None,
            ),
        )
        await db.commit()
    finally:
        await db.close()


@chat_router.post("")
async def chat_stream(req: ChatRequest):
    """SSE endpoint: accept a message, stream agent progress, return analysis."""
    conversation_id = await _ensure_conversation(req.conversation_id, req.message)

    # Save user message
    await _save_message(conversation_id, "user", content=req.message)

    async def event_generator():
        try:
            full_reply = ""
            payload = None

            async for event in analyze_document_streaming(req.message):
                etype = event.get("type")

                if etype == "status":
                    yield f"data: {json.dumps(event)}\n\n"
                elif etype == "agent_done":
                    yield f"data: {json.dumps(event)}\n\n"
                elif etype == "result":
                    payload = event.get("payload", {})
                    full_reply = event.get("full_reply", "")
                    end_event = {
                        "type": "end",
                        "conversation_id": conversation_id,
                        "component": "analysis_card",
                        "payload": payload,
                        "full_reply": full_reply,
                    }
                    yield f"data: {json.dumps(end_event)}\n\n"
                elif etype == "error":
                    yield f"data: {json.dumps(event)}\n\n"

            # Save assistant message
            await _save_message(
                conversation_id,
                "assistant",
                content=full_reply or None,
                component="analysis_card" if payload else None,
                payload=payload,
            )
        except Exception as e:
            logger.exception("Chat stream error")
            err = {"type": "error", "content": str(e)}
            yield f"data: {json.dumps(err)}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@chat_router.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    """Accept .txt or .pdf, extract text, return content."""
    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided")

    ext = file.filename.rsplit(".", 1)[-1].lower() if "." in file.filename else ""

    if ext == "txt":
        raw = await file.read()
        content = raw.decode("utf-8", errors="replace")
    elif ext == "pdf":
        try:
            import pdfplumber
            import io

            raw = await file.read()
            pages_text = []
            with pdfplumber.open(io.BytesIO(raw)) as pdf:
                for page in pdf.pages:
                    text = page.extract_text()
                    if text:
                        pages_text.append(text)
            content = "\n\n".join(pages_text)
        except ImportError:
            raise HTTPException(
                status_code=500,
                detail="PDF support requires pdfplumber. Install it with: pip install pdfplumber",
            )
    else:
        raise HTTPException(
            status_code=400, detail="Unsupported file type. Only .txt and .pdf are accepted."
        )

    return {"filename": file.filename, "content": content}


@chat_router.get("")
async def list_conversations():
    """List all conversations, newest first."""
    db = await get_db()
    try:
        cursor = await db.execute(
            "SELECT id, name, updated_at FROM conversations ORDER BY updated_at DESC"
        )
        rows = await cursor.fetchall()
        return [
            {"id": r["id"], "name": r["name"], "updated_at": r["updated_at"]}
            for r in rows
        ]
    finally:
        await db.close()


@chat_router.get("/{conversation_id}/history")
async def get_history(conversation_id: str):
    """Return all messages for a conversation."""
    db = await get_db()
    try:
        cursor = await db.execute(
            "SELECT id, role, content, component, payload, created_at "
            "FROM chat_messages WHERE conversation_id = ? ORDER BY created_at",
            (conversation_id,),
        )
        rows = await cursor.fetchall()
        messages = []
        for r in rows:
            msg = {
                "id": r["id"],
                "role": r["role"],
                "content": r["content"],
                "component": r["component"],
                "payload": json.loads(r["payload"]) if r["payload"] else None,
                "created_at": r["created_at"],
            }
            messages.append(msg)
        return messages
    finally:
        await db.close()
