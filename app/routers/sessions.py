"""Session management API endpoints."""

from typing import Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from app.services.session_manager import SessionManager

router = APIRouter(prefix="/api/sessions", tags=["sessions"])

_manager: SessionManager | None = None


async def get_session_manager() -> SessionManager:
    """Dependency that returns the singleton SessionManager."""
    global _manager
    if _manager is None:
        _manager = SessionManager()
        await _manager.init_db()
    return _manager


class CreateSessionRequest(BaseModel):
    title: str
    model_id: Optional[str] = None


class UpdateSessionRequest(BaseModel):
    title: Optional[str] = None
    model_id: Optional[str] = None


@router.get("/")
async def list_sessions(sm: SessionManager = Depends(get_session_manager)):
    """List all sessions ordered by most recently updated."""
    return await sm.list_sessions()


@router.post("/", status_code=201)
async def create_session(
    body: CreateSessionRequest,
    sm: SessionManager = Depends(get_session_manager),
):
    """Create a new chat session."""
    return await sm.create_session(title=body.title, model_id=body.model_id)


@router.get("/{session_id}")
async def get_session(
    session_id: str,
    sm: SessionManager = Depends(get_session_manager),
):
    """Get a single session by ID."""
    session = await sm.get_session(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")
    return session


@router.put("/{session_id}")
async def update_session(
    session_id: str,
    body: UpdateSessionRequest,
    sm: SessionManager = Depends(get_session_manager),
):
    """Update session title and/or model."""
    session = await sm.get_session(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")
    await sm.update_session(session_id, title=body.title, model_id=body.model_id)
    return await sm.get_session(session_id)


@router.delete("/{session_id}", status_code=204)
async def delete_session(
    session_id: str,
    sm: SessionManager = Depends(get_session_manager),
):
    """Delete a session and all its messages."""
    await sm.delete_session(session_id)
    return None


@router.get("/{session_id}/messages")
async def get_messages(
    session_id: str,
    sm: SessionManager = Depends(get_session_manager),
):
    """Get all messages for a session."""
    return await sm.get_messages(session_id)


@router.get("/{session_id}/tokens")
async def get_token_count(
    session_id: str,
    sm: SessionManager = Depends(get_session_manager),
):
    """Get total token count for a session."""
    count = await sm.get_session_token_count(session_id)
    return {"session_id": session_id, "token_count": count}
