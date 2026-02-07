import json
import logging
from fastapi import APIRouter, BackgroundTasks, HTTPException

from src.database import get_db
from src.models import DocumentInput, DocumentDetail, DocumentResponse, AnalysisResult
from src.orchestrator import analyze_document

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api")


async def _run_analysis(doc_id: int, content: str):
    """Background task: run the multi-agent analysis pipeline."""
    db = await get_db()
    try:
        await db.execute(
            "UPDATE analyses SET status = 'processing' WHERE document_id = ?",
            (doc_id,),
        )
        await db.commit()

        result = await analyze_document(content)

        combined = result.model_dump()
        await db.execute(
            """UPDATE analyses
               SET summary = ?, actions = ?, risks = ?,
                   combined_output = ?, status = 'completed'
               WHERE document_id = ?""",
            (
                json.dumps({"summary": combined["summary"], "key_decisions": combined["key_decisions"]}),
                json.dumps(combined["action_items"]),
                json.dumps(combined["risks"]),
                json.dumps(combined),
                doc_id,
            ),
        )
        await db.commit()
        logger.info(f"Analysis completed for document {doc_id}")
    except Exception as e:
        logger.exception(f"Analysis failed for document {doc_id}: {e}")
        await db.execute(
            "UPDATE analyses SET status = 'failed' WHERE document_id = ?",
            (doc_id,),
        )
        await db.commit()
    finally:
        await db.close()


@router.post("/documents")
async def create_document(doc: DocumentInput, background_tasks: BackgroundTasks):
    word_count = len(doc.content.split())
    db = await get_db()
    try:
        cursor = await db.execute(
            "INSERT INTO documents (title, content, word_count) VALUES (?, ?, ?)",
            (doc.title, doc.content, word_count),
        )
        doc_id = cursor.lastrowid
        await db.execute(
            "INSERT INTO analyses (document_id) VALUES (?)",
            (doc_id,),
        )
        await db.commit()
    finally:
        await db.close()

    background_tasks.add_task(_run_analysis, doc_id, doc.content)
    return {"id": doc_id, "status": "processing", "word_count": word_count}


@router.get("/documents")
async def list_documents():
    db = await get_db()
    try:
        cursor = await db.execute(
            "SELECT d.id, d.title, d.word_count, d.created_at, a.status "
            "FROM documents d LEFT JOIN analyses a ON d.id = a.document_id "
            "ORDER BY d.created_at DESC"
        )
        rows = await cursor.fetchall()
        return [
            {
                "id": r["id"],
                "title": r["title"],
                "word_count": r["word_count"],
                "created_at": r["created_at"],
                "analysis_status": r["status"],
            }
            for r in rows
        ]
    finally:
        await db.close()


@router.get("/documents/{doc_id}")
async def get_document(doc_id: int):
    db = await get_db()
    try:
        cursor = await db.execute(
            "SELECT d.*, a.combined_output, a.status AS analysis_status "
            "FROM documents d LEFT JOIN analyses a ON d.id = a.document_id "
            "WHERE d.id = ?",
            (doc_id,),
        )
        row = await cursor.fetchone()
        if not row:
            raise HTTPException(status_code=404, detail="Document not found")

        analysis = None
        if row["combined_output"]:
            try:
                analysis = json.loads(row["combined_output"])
            except json.JSONDecodeError:
                pass

        return {
            "id": row["id"],
            "title": row["title"],
            "content": row["content"],
            "word_count": row["word_count"],
            "created_at": row["created_at"],
            "analysis_status": row["analysis_status"],
            "analysis": analysis,
        }
    finally:
        await db.close()


@router.get("/analyses/{doc_id}")
async def get_analysis(doc_id: int):
    db = await get_db()
    try:
        cursor = await db.execute(
            "SELECT * FROM analyses WHERE document_id = ?", (doc_id,)
        )
        row = await cursor.fetchone()
        if not row:
            raise HTTPException(status_code=404, detail="Analysis not found")

        result = {
            "id": row["id"],
            "document_id": row["document_id"],
            "status": row["status"],
            "created_at": row["created_at"],
        }
        if row["combined_output"]:
            try:
                result["result"] = json.loads(row["combined_output"])
            except json.JSONDecodeError:
                result["result"] = None
        return result
    finally:
        await db.close()
