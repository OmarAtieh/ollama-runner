"""Session and project management API endpoints."""

from typing import Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from app.services.session_manager import SessionManager

router = APIRouter(prefix="/api/sessions", tags=["sessions"])
projects_router = APIRouter(prefix="/api/projects", tags=["projects"])

_manager: SessionManager | None = None


async def get_session_manager() -> SessionManager:
    """Dependency that returns the singleton SessionManager."""
    global _manager
    if _manager is None:
        _manager = SessionManager()
        await _manager.init_db()
    return _manager


# ── Session request models ────────────────────────────────────


class CreateSessionRequest(BaseModel):
    title: str
    model_id: Optional[str] = None
    project_id: Optional[str] = None


class UpdateSessionRequest(BaseModel):
    title: Optional[str] = None
    model_id: Optional[str] = None
    project_id: Optional[str] = "__unset__"


# ── Project request models ────────────────────────────────────


class CreateProjectRequest(BaseModel):
    name: str
    color: str = "#6c8cff"
    system_prompt: str = ""


class UpdateProjectRequest(BaseModel):
    name: Optional[str] = None
    color: Optional[str] = None
    system_prompt: Optional[str] = None


# ── Session endpoints ─────────────────────────────────────────


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
    return await sm.create_session(
        title=body.title, model_id=body.model_id, project_id=body.project_id
    )


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
    """Update session title, model, and/or project."""
    session = await sm.get_session(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")
    await sm.update_session(
        session_id,
        title=body.title,
        model_id=body.model_id,
        project_id=body.project_id,
    )
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


# ── Project endpoints ─────────────────────────────────────────


@projects_router.get("/")
async def list_projects(sm: SessionManager = Depends(get_session_manager)):
    """List all projects."""
    return await sm.list_projects()


@projects_router.post("/", status_code=201)
async def create_project(
    body: CreateProjectRequest,
    sm: SessionManager = Depends(get_session_manager),
):
    """Create a new project folder."""
    return await sm.create_project(
        name=body.name, color=body.color, system_prompt=body.system_prompt
    )


@projects_router.put("/{project_id}")
async def update_project(
    project_id: str,
    body: UpdateProjectRequest,
    sm: SessionManager = Depends(get_session_manager),
):
    """Update a project."""
    result = await sm.update_project(
        project_id, name=body.name, color=body.color, system_prompt=body.system_prompt
    )
    if result is None:
        raise HTTPException(status_code=404, detail="Project not found")
    return result


@projects_router.delete("/{project_id}", status_code=204)
async def delete_project(
    project_id: str,
    sm: SessionManager = Depends(get_session_manager),
):
    """Delete a project. Sessions are moved to Unsorted."""
    await sm.delete_project(project_id)
    return None
