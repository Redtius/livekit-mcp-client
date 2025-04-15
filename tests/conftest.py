import pytest
from unittest.mock import MagicMock, AsyncMock

try:
    from mcp import ClientSession
except ImportError:

    class ClientSessionSpec:
        async def __aenter__(self):
            pass

        async def __aexit__(self, exc_type, exc_val, exc_tb):
            pass

        async def initialize(self):
            pass

        async def send_ping(self):
            pass

        async def list_tools(self):
            pass

        async def call_tool(self, tool_name, kwargs):
            pass

    ClientSession = ClientSessionSpec
    print("Warning: Failed to import real ClientSession, using fallback spec.")


@pytest.fixture(scope="function")
def mock_shared_session():
    """Provides a shared, configured mock ClientSession."""
    print("\n--- Creating shared mock session ---")

    session = MagicMock(spec=ClientSession, name="SharedMockClientSession")

    session.__aenter__ = AsyncMock(return_value=session)
    session.__aexit__ = AsyncMock(return_value=None)
    session.initialize = AsyncMock(name="initialize_async_mock")
    session.send_ping = AsyncMock(name="send_ping_async_mock")
    session.list_tools = AsyncMock(name="list_tools_async_mock")
    session.call_tool = AsyncMock(name="call_tool_async_mock")

    session.reset_mock()

    yield session

    print("--- Tearing down shared mock session ---")
