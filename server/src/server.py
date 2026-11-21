# -*- coding: utf-8 -*-
"""
Agora Agent & Token Service

HTTP APIs:
- GET  /get_config     -> Agent.generate_config()
- POST /startAgent     -> Agent.start()
- POST /stopAgent      -> Agent.stop()
"""
import logging
import os
import random
import time
from typing import Any, Dict, List, Optional
from dotenv import load_dotenv

# The Agora CLI writes the Python quickstart environment to server/.env.
_base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
load_dotenv(os.path.join(_base_dir, '.env'), override=True)

from fastapi import APIRouter, FastAPI, Header, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from agora_agent.agentkit.token import generate_convo_ai_token
from agent import Agent

logger = logging.getLogger("uvicorn.error")


def _log_route_error(route: str, exc: Exception, **context) -> None:
    """Log route failures with safe request context and a traceback."""
    safe_context = {key: value for key, value in context.items() if value is not None}
    logger.exception(
        "Request failed route=%s context=%s error_type=%s error=%s",
        route,
        safe_context,
        type(exc).__name__,
        exc,
    )


def _to_http_error(exc: Exception) -> HTTPException:
    """Convert SDK exceptions to HTTP errors"""
    if isinstance(exc, ValueError):
        return HTTPException(status_code=400, detail=str(exc))
    if isinstance(exc, RuntimeError):
        return HTTPException(status_code=500, detail=str(exc))
    return HTTPException(status_code=500, detail=f"Internal error: {exc}")

try:
    agent = Agent()
except ValueError as e:
    logger.exception(
        "Failed to initialize Agora Agent SDK. Service will fail if endpoints are called without proper configuration: %s",
        e,
    )
    agent = None


# FastAPI application
app = FastAPI(
    title="Agora Agent & Token Service",
    version="2.0.0",
    description="Agora Conversational AI service",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

router = APIRouter()


# Request models
class StartAgentRequest(BaseModel):
    """Request body for POST /startAgent"""
    channelName: str
    rtcUid: int
    userUid: int
    familyId: Optional[str] = None
    memberId: Optional[str] = None
    memberName: Optional[str] = None
    memberRole: Optional[str] = None
    familyName: Optional[str] = None
    timezone: Optional[str] = None
    parameters: Optional[Dict[str, Any]] = None


class StopAgentRequest(BaseModel):
    """Request body for POST /stopAgent"""
    agentId: str


class ToolSearchRequest(BaseModel):
    query: str


class ToolRememberRequest(BaseModel):
    content: str
    visibility: str = "private"


class SyncMessage(BaseModel):
    role: str = "user"
    text: str


class SyncConversationRequest(BaseModel):
    familyId: str
    memberId: str
    channelName: str
    startedAt: Optional[str] = None
    messages: List[SyncMessage]


# API endpoints
def _generate_channel_name() -> str:
    return f"ai-conversation-{int(time.time())}-{random.randint(1000, 9999)}"


@router.get("/get_config")
async def get_config(
    channel: Optional[str] = Query(default=None),
    uid: Optional[int] = Query(default=None),
):
    """Generate connection configuration"""
    if agent is None:
        raise HTTPException(
            status_code=500,
            detail="Service not properly configured. Please check environment variables.",
        )

    try:
        # Agora RTC accepts uid=0 as "auto assign", but RTM token subjects cannot
        # use 0. Replace missing, zero, or negative values with a generated UID.
        user_uid = random.randint(1000, 9999999) if uid is None or uid <= 0 else uid
        agent_uid = str(random.randint(10000000, 99999999))
        channel_name = channel or _generate_channel_name()

        # Get credentials from environment
        app_id = os.getenv("AGORA_APP_ID")
        app_certificate = os.getenv("AGORA_APP_CERTIFICATE")

        # Generate a one-hour RTC+RTM token and renew it client-side as needed.
        token = generate_convo_ai_token(
            app_id=app_id,
            app_certificate=app_certificate,
            channel_name=channel_name,
            uid=user_uid,
            token_expire=3600,
        )

        config_data = {
            "app_id": app_id,
            "token": token,
            "uid": str(user_uid),
            "channel_name": channel_name,
            "agent_uid": agent_uid,
        }

        return {
            "code": 0,
            "data": config_data,
            "msg": "success",
        }
    except Exception as e:
        _log_route_error("/get_config", e, channel=channel, uid=uid)
        raise _to_http_error(e)


@router.post("/startAgent")
async def start_agent(request: StartAgentRequest):
    """Start agent in a channel"""
    if agent is None:
        raise HTTPException(
            status_code=500,
            detail="Service not properly configured. Please check environment variables.",
        )

    try:
        output_audio_codec = None
        if request.parameters:
            output_audio_codec = request.parameters.get("output_audio_codec")

        result = await agent.start(
            channel_name=request.channelName,
            agent_uid=request.rtcUid,
            user_uid=request.userUid,
            output_audio_codec=output_audio_codec,
            family_id=request.familyId,
            member_id=request.memberId,
            member_name=request.memberName,
            member_role=request.memberRole,
            family_name=request.familyName,
            timezone=request.timezone,
        )
        return {"code": 0, "msg": "success", "data": result}
    except Exception as e:
        _log_route_error(
            "/startAgent",
            e,
            channelName=request.channelName,
            rtcUid=request.rtcUid,
            userUid=request.userUid,
        )
        raise _to_http_error(e)


@router.post("/stopAgent")
async def stop_agent(request: StopAgentRequest):
    """Stop agent by ID"""
    if agent is None:
        raise HTTPException(
            status_code=500,
            detail="Service not properly configured. Please check environment variables.",
        )

    try:
        await agent.stop(request.agentId)
        return {"code": 0, "msg": "success"}
    except Exception as e:
        _log_route_error("/stopAgent", e, agentId=request.agentId)
        raise _to_http_error(e)


def _resolve_memory_session(token: Optional[str]):
    if agent is None or not token or not hasattr(agent, "memory"):
        return None
    return agent.memory.resolve_tool_session(token)


@router.post("/tools/memory/search")
async def tool_memory_search(
    request: ToolSearchRequest,
    x_memory_session: Optional[str] = Header(default=None),
):
    """Inline REST tool endpoint called by the Agora agent cloud."""
    resolved = _resolve_memory_session(x_memory_session)
    if resolved is None:
        return {"result": "Memory is unavailable for this session."}

    family_id, member_id = resolved
    result = await agent.memory.search_text(family_id, member_id, request.query)
    return {"result": result}


@router.post("/tools/memory/add")
async def tool_memory_add(
    request: ToolRememberRequest,
    x_memory_session: Optional[str] = Header(default=None),
):
    """Inline REST tool endpoint called by the Agora agent cloud."""
    resolved = _resolve_memory_session(x_memory_session)
    if resolved is None:
        return {"result": "Memory is unavailable for this session."}

    family_id, member_id = resolved
    visibility = "family" if request.visibility == "family" else "private"
    ok = await agent.memory.remember(family_id, member_id, request.content, visibility)
    return {"result": "Saved." if ok else "I could not save that right now."}


@router.post("/memory/sync")
async def sync_memory(request: SyncConversationRequest):
    """Persist one completed voice conversation to Supermemory."""
    if agent is None or not hasattr(agent, "memory"):
        raise HTTPException(
            status_code=500,
            detail="Service not properly configured. Please check environment variables.",
        )

    ok = await agent.memory.sync_conversation(
        family_id=request.familyId,
        member_id=request.memberId,
        channel_name=request.channelName,
        messages=[m.model_dump() for m in request.messages],
        started_at=request.startedAt,
    )
    return {"code": 0, "msg": "success", "data": {"saved": ok}}


app.include_router(router)


if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("PORT", "8000"))
    uvicorn.run(app, host="0.0.0.0", port=port)
