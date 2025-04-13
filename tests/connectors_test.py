import pytest
import asyncio
from unittest.mock import patch, MagicMock, AsyncMock, ANY
from contextlib import asynccontextmanager

from livekit_mcp_client.connectors.sse import SSEConnector
from livekit_mcp_client.connectors.stdio import StdioConnector

@asynccontextmanager
async def mock_sse_client(*args, **kwargs):
    yield (AsyncMock(name="sse_read_stream"), AsyncMock(name="sse_write_stream"))

@asynccontextmanager
async def mock_stdio_client(*args, **kwargs):
     yield (AsyncMock(name="stdio_read_stream"), AsyncMock(name="stdio_write_stream"))


# --- Test SSEConnector ---

@pytest.fixture
def sse_connector() -> SSEConnector:
    return SSEConnector(url="http://fake.url")

@pytest.mark.asyncio
@patch('livekit_mcp_client.connectors.sse.sse_client', new=mock_sse_client)
@patch('livekit_mcp_client.connectors.sse.ClientSession')
async def test_sse_connect(mock_ClientSession: MagicMock,
                           sse_connector: SSEConnector,
                           mock_shared_session: MagicMock):
    """Test successful SSE connection."""
    mock_ClientSession.return_value = mock_shared_session

    sse_connector._session = None

    # --- Action ---
    await sse_connector.connect()

    # --- Assertions ---
    mock_ClientSession.assert_called_once()
    mock_shared_session.__aenter__.assert_awaited_once()
    mock_shared_session.initialize.assert_awaited_once()

    assert sse_connector._session is mock_shared_session
    assert sse_connector.get_session() is mock_shared_session

@pytest.mark.asyncio
async def test_sse_disconnect(sse_connector: SSEConnector):
    """Test disconnection."""
    sse_connector._exit_stack = AsyncMock()
    sse_connector._session = MagicMock(name="disconnect_session_mock")

    await sse_connector.disconnect()

    sse_connector._exit_stack.aclose.assert_awaited_once()
    assert sse_connector._session is None

@pytest.mark.asyncio
async def test_sse_is_alive_no_session(sse_connector: SSEConnector):
    """Test is_alive when not connected."""
    sse_connector._session = None
    assert await sse_connector.is_alive() is False

@pytest.mark.asyncio
async def test_sse_is_alive_success(sse_connector: SSEConnector, mock_shared_session: MagicMock):
    """Test is_alive when connection is healthy."""
    sse_connector._session = mock_shared_session
    mock_shared_session.send_ping.side_effect = None
    mock_shared_session.send_ping.return_value = None

    assert await sse_connector.is_alive() is True
    mock_shared_session.send_ping.assert_awaited_once()

@pytest.mark.asyncio
async def test_sse_is_alive_timeout(sse_connector: SSEConnector, mock_shared_session: MagicMock):
    """Test is_alive when send_ping times out."""
    sse_connector._session = mock_shared_session
    mock_shared_session.send_ping.side_effect = asyncio.TimeoutError

    assert await sse_connector.is_alive() is False
    mock_shared_session.send_ping.assert_awaited_once()

@pytest.mark.asyncio
async def test_sse_is_alive_connection_error(sse_connector: SSEConnector, mock_shared_session: MagicMock):
    """Test is_alive when send_ping raises ConnectionError."""
    sse_connector._session = mock_shared_session
    mock_shared_session.send_ping.side_effect = ConnectionError

    assert await sse_connector.is_alive() is False
    mock_shared_session.send_ping.assert_awaited_once()


# --- Test StdioConnector ---

@pytest.fixture
def stdio_connector() -> StdioConnector:
    return StdioConnector(command="mycmd", args=["arg1"], env={"K": "V"})

@pytest.mark.asyncio
@patch('livekit_mcp_client.connectors.stdio.stdio_client', new=mock_stdio_client)
@patch('livekit_mcp_client.connectors.stdio.ClientSession')
async def test_stdio_connect(mock_ClientSession: MagicMock,
                             stdio_connector: StdioConnector,
                             mock_shared_session: MagicMock):
    """Test successful Stdio connection."""
    mock_ClientSession.return_value = mock_shared_session

    stdio_connector._session = None

    # --- Action ---
    await stdio_connector.connect()

    # --- Assertions ---
    mock_ClientSession.assert_called_once()
    # mock_ClientSession.assert_called_once_with(ANY, ANY)

    mock_shared_session.__aenter__.assert_awaited_once()
    mock_shared_session.initialize.assert_awaited_once()

    assert stdio_connector._session is mock_shared_session
    assert stdio_connector.get_session() is mock_shared_session

@pytest.mark.asyncio
async def test_stdio_disconnect(stdio_connector: StdioConnector):
    """Test disconnection."""
    stdio_connector._exit_stack = AsyncMock()
    stdio_connector._session = MagicMock(name="disconnect_session_mock")

    await stdio_connector.disconnect()

    stdio_connector._exit_stack.aclose.assert_awaited_once()
    assert stdio_connector._session is None

@pytest.mark.asyncio
async def test_stdio_is_alive_no_session(stdio_connector: StdioConnector):
    """Test is_alive when not connected."""
    stdio_connector._session = None
    assert await stdio_connector.is_alive() is False

@pytest.mark.asyncio
async def test_stdio_is_alive_success(stdio_connector: StdioConnector, mock_shared_session: MagicMock):
    """Test is_alive when connection is healthy."""
    stdio_connector._session = mock_shared_session
    mock_shared_session.list_tools.side_effect = None
    mock_shared_session.list_tools.return_value = MagicMock()

    assert await stdio_connector.is_alive() is True
    mock_shared_session.list_tools.assert_awaited_once()

@pytest.mark.asyncio
async def test_stdio_is_alive_timeout(stdio_connector: StdioConnector, mock_shared_session: MagicMock):
    """Test is_alive when list_tools times out."""
    stdio_connector._session = mock_shared_session
    mock_shared_session.list_tools.side_effect = asyncio.TimeoutError

    assert await stdio_connector.is_alive() is False
    mock_shared_session.list_tools.assert_awaited_once()

@pytest.mark.asyncio
async def test_stdio_is_alive_connection_error(stdio_connector: StdioConnector, mock_shared_session: MagicMock):
    """Test is_alive when list_tools raises ConnectionError."""
    stdio_connector._session = mock_shared_session
    mock_shared_session.list_tools.side_effect = ConnectionError

    assert await stdio_connector.is_alive() is False
    mock_shared_session.list_tools.assert_awaited_once()