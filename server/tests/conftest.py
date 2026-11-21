"""Shared fixtures for the server test suite.

Standalone: no Agora cloud, no real credentials. A deterministic fake env is
injected, and python-dotenv is neutralized so a developer's real `server/.env`
cannot override the test env (server.py loads it with override=True).
"""
import importlib
import os
import sys

import pytest

# Make `import server` / `import agent` resolve to server/src/*.
_SERVER_SRC = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
if _SERVER_SRC not in sys.path:
    sys.path.insert(0, _SERVER_SRC)

FAKE_ENV = {
    "AGORA_APP_ID": "0123456789abcdef0123456789abcdef",
    "AGORA_APP_CERTIFICATE": "fedcba9876543210fedcba9876543210",
}


@pytest.fixture
def fake_env(monkeypatch):
    """Inject a deterministic env and stop dotenv from clobbering it."""
    import dotenv

    monkeypatch.setattr(dotenv, "load_dotenv", lambda *a, **k: False)
    for key, value in FAKE_ENV.items():
        monkeypatch.setenv(key, value)
    return dict(FAKE_ENV)


class FakeMemory:
    async def search_text(self, family_id, member_id, query):
        return "No matching memories."

    async def remember(self, family_id, member_id, content, visibility="private"):
        return True

    async def sync_conversation(self, family_id, member_id, channel_name, messages, started_at=None):
        return True

    def resolve_tool_session(self, token):
        return ("default-family", "default-member") if token else None


class FakeAgent:
    """Stand-in for the real Agent (mirrors scripts/run_fake_server.py)."""

    def __init__(self):
        self.started = []
        self.stopped = []
        self.memory = FakeMemory()

    async def start(self, channel_name, agent_uid, user_uid, output_audio_codec=None, **kwargs):
        self.started.append((channel_name, agent_uid, user_uid, output_audio_codec, kwargs))
        return {
            "agent_id": f"fake-agent-{agent_uid}",
            "channel_name": channel_name,
            "status": "started",
            "family_id": kwargs.get("family_id") or "default-family",
            "member_id": kwargs.get("member_id") or str(user_uid),
        }

    async def stop(self, agent_id):
        self.stopped.append(agent_id)


@pytest.fixture
def server_module(fake_env):
    """Import server.py fresh, with the fake env + neutralized dotenv applied."""
    sys.modules.pop("server", None)
    sys.modules.pop("agent", None)
    import server

    importlib.reload(server)
    return server


@pytest.fixture
def client(server_module):
    """A FastAPI TestClient whose agent is a FakeAgent (no cloud)."""
    from fastapi.testclient import TestClient

    fake = FakeAgent()
    server_module.agent = fake
    test_client = TestClient(server_module.app)
    test_client.fake_agent = fake
    return test_client
