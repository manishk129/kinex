"""Supermemory-backed memory layer for Family OS.

The local Supermemory server (`supermemory-server`) speaks the same API as the
hosted platform. Point the SDK at it with:

    SUPERMEMORY_BASE_URL=http://localhost:6767
    SUPERMEMORY_API_KEY=sm_...

Namespaces are the privacy boundary:

    family:<family_id>:shared             # visible to every member
    family:<family_id>:member:<member_id> # visible only to that member

The layer has three jobs:

1. Bootstrap the system prompt with the speaker's profile, recent memories,
   and shared family context.
2. Expose `search_memory` / `remember` as inline REST tools while a call runs.
3. Persist the conversation transcript when the call ends.

Every operation degrades gracefully: a voice session must still start when
Supermemory is unreachable.
"""
from __future__ import annotations

import asyncio
import logging
import os
import secrets
import time
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger("uvicorn.error")

try:  # The memory layer is optional at import time.
    from supermemory import AsyncSupermemory
except Exception:  # pragma: no cover - exercised only when the SDK is absent
    AsyncSupermemory = None  # type: ignore[assignment]


MEMORY_SESSION_TTL_SECONDS = 2 * 60 * 60
MAX_PROFILE_LINES = 20
MAX_SEARCH_RESULTS = 6
DEFAULT_THRESHOLD = 0.6


def _env(name: str, default: str = "") -> str:
    return (os.getenv(name) or default).strip()


def family_namespace(family_id: str) -> str:
    return f"family:{family_id}:shared"


def member_namespace(family_id: str, member_id: str) -> str:
    return f"family:{family_id}:member:{member_id}"


def _format_lines(lines: List[str], empty: str) -> str:
    """Return a short, deduped bullet list for prompt injection."""
    seen: List[str] = []
    for line in lines:
        text = " ".join((line or "").split())
        if text and text not in seen:
            seen.append(text)
        if len(seen) >= MAX_PROFILE_LINES:
            break
    if not seen:
        return empty
    return "\n".join(f"- {line}" for line in seen)


class MemoryStore:
    """Thin async wrapper around Supermemory plus Family OS privacy scoping."""

    def __init__(self) -> None:
        self.api_key = _env("SUPERMEMORY_API_KEY")
        self.base_url = _env("SUPERMEMORY_BASE_URL", "http://localhost:6767")
        self._tool_sessions: Dict[str, Tuple[str, str, float]] = {}
        self.client: Any = None

        if self.api_key and AsyncSupermemory is not None:
            try:
                self.client = AsyncSupermemory(
                    api_key=self.api_key,
                    base_url=self.base_url,
                    timeout=10.0,
                )
            except Exception:  # pragma: no cover - construction is normally safe
                logger.exception("Failed to initialize the Supermemory client")
        else:
            logger.warning(
                "Supermemory memory layer disabled (missing SUPERMEMORY_API_KEY or supermemory SDK)"
            )

    @property
    def enabled(self) -> bool:
        return self.client is not None

    # ------------------------------------------------------------------
    # Read paths
    # ------------------------------------------------------------------

    async def _profile_facts(self, namespace: str) -> Tuple[List[str], List[str]]:
        if self.client is None:
            return [], []
        try:
            response = await self.client.profile(namespace)
            profile = response.profile
            static = [
                fact.memory for fact in profile.static if getattr(fact, "memory", None)
            ]
            dynamic = [
                fact.memory for fact in profile.dynamic if getattr(fact, "memory", None)
            ]
            return static, dynamic
        except Exception:
            logger.warning(
                "Supermemory profile lookup failed namespace=%s",
                namespace,
                exc_info=True,
            )
            return [], []

    async def context_for(self, family_id: str, member_id: str) -> Dict[str, str]:
        """Build the three memory blocks used by FAMILY_OS_PROMPT."""
        if self.client is None:
            return {
                "member_profile": "No profile yet.",
                "memories": "No recent memories.",
                "family_context": "No shared context yet.",
            }

        member_result, family_result = await asyncio.gather(
            self._profile_facts(member_namespace(family_id, member_id)),
            self._profile_facts(family_namespace(family_id)),
        )
        member_static, member_dynamic = member_result
        family_static, family_dynamic = family_result

        return {
            "member_profile": _format_lines(member_static, "No profile yet."),
            "memories": _format_lines(member_dynamic, "No recent memories."),
            "family_context": _format_lines(
                family_static + family_dynamic, "No shared context yet."
            ),
        }

    async def search_text(
        self,
        family_id: str,
        member_id: str,
        query: str,
        *,
        limit: int = MAX_SEARCH_RESULTS,
        threshold: float = DEFAULT_THRESHOLD,
    ) -> str:
        """Search private + shared memory and return a voice-friendly block."""
        query = (query or "").strip()
        if self.client is None or not query:
            return "No matching memories."

        try:
            private, shared = await asyncio.gather(
                self.client.search(
                    member_namespace(family_id, member_id),
                    query=query,
                    search_mode="memories",
                    limit=limit,
                    threshold=threshold,
                ),
                self.client.search(
                    family_namespace(family_id),
                    query=query,
                    search_mode="memories",
                    limit=max(2, limit // 2),
                    threshold=threshold,
                ),
            )
        except Exception:
            logger.warning(
                "Supermemory search failed family=%s member=%s",
                family_id,
                member_id,
                exc_info=True,
            )
            return "Memory search is unavailable right now."

        lines: List[str] = []
        for response in (private, shared):
            for result in response.results:
                text = result.memory or result.chunk
                if text:
                    lines.append(text)

        return _format_lines(lines, "No matching memories.")

    # ------------------------------------------------------------------
    # Write paths
    # ------------------------------------------------------------------

    async def add(
        self,
        namespace: str,
        content: str,
        *,
        metadata: Optional[Dict[str, Any]] = None,
        document_id: Optional[str] = None,
    ) -> bool:
        content = (content or "").strip()
        if self.client is None or not content:
            return False

        try:
            await self.client.add(
                namespace,
                content=content,
                metadata=metadata or {},
                id=document_id,
                dreaming="instant",
            )
            return True
        except Exception:
            logger.warning(
                "Supermemory add failed namespace=%s document_id=%s",
                namespace,
                document_id,
                exc_info=True,
            )
            return False

    async def remember(
        self,
        family_id: str,
        member_id: str,
        content: str,
        visibility: str = "private",
    ) -> bool:
        """Save a fact to the private or shared namespace.

        `visibility` is validated here, not trusted from the model. Anything
        other than an explicit "family" write stays private.
        """
        namespace = (
            family_namespace(family_id)
            if visibility == "family"
            else member_namespace(family_id, member_id)
        )
        return await self.add(
            namespace,
            content,
            metadata={
                "family_id": family_id,
                "member_id": member_id,
                "visibility": "family" if visibility == "family" else "private",
                "source": "voice",
            },
        )

    async def sync_conversation(
        self,
        family_id: str,
        member_id: str,
        channel_name: str,
        messages: List[Dict[str, str]],
        *,
        started_at: Optional[str] = None,
    ) -> bool:
        """Persist one conversation transcript to the speaker's namespace."""
        lines: List[str] = []
        for message in messages:
            role = (message.get("role") or "user").strip()
            text = (message.get("text") or "").strip()
            if text:
                lines.append(f"{role}: {text}")

        if not lines:
            return False

        return await self.add(
            member_namespace(family_id, member_id),
            "\n".join(lines),
            metadata={
                "family_id": family_id,
                "member_id": member_id,
                "visibility": "private",
                "source": "voice-transcript",
                "channel": channel_name,
                "started_at": started_at or "",
            },
            document_id=f"conversation:{channel_name}",
        )

    # ------------------------------------------------------------------
    # Tool sessions
    # ------------------------------------------------------------------

    def _prune_tool_sessions(self) -> None:
        now = time.time()
        expired = [
            token
            for token, (_, _, expires_at) in self._tool_sessions.items()
            if expires_at <= now
        ]
        for token in expired:
            self._tool_sessions.pop(token, None)

    def create_tool_session(self, family_id: str, member_id: str) -> str:
        """Mint an opaque token that binds tool calls to one speaker."""
        self._prune_tool_sessions()
        token = secrets.token_urlsafe(24)
        self._tool_sessions[token] = (
            family_id,
            member_id,
            time.time() + MEMORY_SESSION_TTL_SECONDS,
        )
        return token

    def resolve_tool_session(self, token: str) -> Optional[Tuple[str, str]]:
        """Resolve a tool token to `(family_id, member_id)`.

        The model never sees or controls the namespace; it only echoes the
        token that Agora injects from `template_variables`.
        """
        self._prune_tool_sessions()
        entry = self._tool_sessions.get(token)
        if entry is None:
            return None
        family_id, member_id, expires_at = entry
        if expires_at <= time.time():
            self._tool_sessions.pop(token, None)
            return None
        return family_id, member_id

    def tool_definitions(self, public_base_url: str) -> List[Any]:
        """Build Agora inline REST tool definitions for this session."""
        base = (public_base_url or "").rstrip("/")
        if not base:
            return []

        try:
            from urllib.parse import urlparse

            host = (urlparse(base).hostname or "").lower()
            if host in {"localhost", "127.0.0.1", "0.0.0.0", "::1"} or host.endswith(".local"):
                logger.warning(
                    "Memory tools disabled: PUBLIC_BASE_URL=%s is not reachable by Agora cloud. "
                    "Use a public tunnel/deployed URL to enable live memory tools.",
                    base,
                )
                return []
        except Exception:
            logger.warning("Memory tools disabled: invalid PUBLIC_BASE_URL=%s", base)
            return []

        try:
            from agora_agent.agentkit import (
                LlmToolConfig,
                LlmToolExecutionConfig,
                LlmToolFunctionConfig,
                LlmToolFunctionParametersConfig,
                LlmToolServerConfig,
            )
        except Exception:  # pragma: no cover - SDK mismatch
            logger.warning("Agora tool configs unavailable; memory tools disabled")
            return []

        headers = {"X-Memory-Session": "{{template_variables.memory_session}}"}

        return [
            LlmToolConfig(
                type="function",
                function=LlmToolFunctionConfig(
                    name="search_memory",
                    description=(
                        "Search this speaker's private memories and the family's shared "
                        "memories. Use it before answering questions about preferences, "
                        "plans, commitments, or things people told you earlier."
                    ),
                    parameters=LlmToolFunctionParametersConfig(
                        type="object",
                        properties={
                            "query": {
                                "type": "string",
                                "description": "What to look up in memory.",
                            }
                        },
                        required=["query"],
                    ),
                ),
                execution=LlmToolExecutionConfig(mode="sync"),
                server=LlmToolServerConfig(
                    method="POST",
                    url=f"{base}/tools/memory/search",
                    headers=headers,
                    body={"query": "{{args.query}}"},
                    timeout_ms=4000,
                ),
            ),
            LlmToolConfig(
                type="function",
                function=LlmToolFunctionConfig(
                    name="remember",
                    description=(
                        "Save a preference, decision, promise, or important date. "
                        "Use visibility 'private' for this speaker only, or 'family' "
                        "for something the whole family should know."
                    ),
                    parameters=LlmToolFunctionParametersConfig(
                        type="object",
                        properties={
                            "content": {
                                "type": "string",
                                "description": "The fact to remember, written as one sentence.",
                            },
                            "visibility": {
                                "type": "string",
                                "enum": ["private", "family"],
                                "description": "Who should be able to recall this.",
                            },
                        },
                        required=["content", "visibility"],
                    ),
                ),
                execution=LlmToolExecutionConfig(mode="sync"),
                server=LlmToolServerConfig(
                    method="POST",
                    url=f"{base}/tools/memory/add",
                    headers=headers,
                    body={
                        "content": "{{args.content}}",
                        "visibility": "{{args.visibility}}",
                    },
                    timeout_ms=4000,
                ),
            ),
        ]
