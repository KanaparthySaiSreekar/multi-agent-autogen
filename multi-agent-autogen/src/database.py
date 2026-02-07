import os
import aiosqlite

DB_PATH = os.path.join(os.path.dirname(__file__), "..", "db", "documents.db")
SCHEMA_PATH = os.path.join(os.path.dirname(__file__), "..", "db", "schema.sql")


async def get_db() -> aiosqlite.Connection:
    db = await aiosqlite.connect(DB_PATH)
    db.row_factory = aiosqlite.Row
    await db.execute("PRAGMA journal_mode=WAL")
    return db


async def init_db():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    with open(SCHEMA_PATH) as f:
        schema = f.read()
    db = await get_db()
    try:
        await db.executescript(schema)
        await db.commit()
    finally:
        await db.close()
