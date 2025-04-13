import pytest
import asyncio
from unittest.mock import patch, MagicMock, AsyncMock
from contextlib import AsyncExitStack
import logging
from typing import Iterator
import builtins
import inspect

from livekit.agents.llm import FunctionContext
try:
    from mcp import ClientSession
except ImportError:
    ClientSession = MagicMock

from livekit_mcp_client.connectors.base import BaseConnector
from livekit_mcp_client.clients.schema import SchemaReader
from livekit_mcp_client.clients.mcpclient import MCPClient
from livekit_mcp_client import utils
from livekit_mcp_client.exceptions import (
    MCPConnectionTimeoutError,
    MCPConnectionError,
)
# try:
#     from livekit_mcp_client.clients.mcpclient import safe_globals
# except ImportError:
#     safe_globals = {'__builtins__': builtins}


@pytest.fixture
def mock_shared_session():
    """Provides a mock ClientSession."""
    print("\n--- Creating shared mock session ---")
    session = MagicMock(name='SharedMockClientSession', spec=ClientSession)
    session.connect = AsyncMock(name="session_connect")
    session.disconnect = AsyncMock(name="session_disconnect")
    session.send_ping = AsyncMock(name="session_send_ping")
    session.list_tools = AsyncMock(name="session_list_tools")
    session.call_tool = AsyncMock(name="session_call_tool")
    yield session
    print("--- Tearing down shared mock session ---")

@pytest.fixture
def mock_connector(mock_shared_session: MagicMock):
    """Provides a mock BaseConnector using the shared session."""
    connector = MagicMock(spec=BaseConnector)
    connector.connect = AsyncMock(name="connector_connect")
    connector.disconnect = AsyncMock(name="connector_disconnect")
    connector.is_alive = AsyncMock(name="connector_is_alive", return_value=True)
    connector.get_session = MagicMock(name="connector_get_session", return_value=mock_shared_session)
    return connector

@pytest.fixture
def mcp_client(mock_connector: MagicMock) -> Iterator[MCPClient]:
    """Provides an MCPClient instance with a mocked connector."""
    client = MCPClient(connector=mock_connector, heartbeat_interval=999)
    client._heartbeat_task = None
    yield client
    if client._is_alive or client._heartbeat_task:
        try:
            loop = asyncio.get_event_loop_policy().get_event_loop()
            if loop.is_running():
                 asyncio.ensure_future(client.aclose()) 
            else:
                 loop.run_until_complete(client.aclose())
        except Exception as e:
             print(f"Error during mcp_client fixture teardown: {e}")


# --- Mock Implementations ---
def mock_ai_callable_impl(**decorator_kwargs):
    def decorator(func):
        func._ai_callable_kwargs = decorator_kwargs
        if not asyncio.iscoroutinefunction(func):
             async def async_wrapper(*args, **kwargs):
                 return func(*args, **kwargs)
             async_wrapper._ai_callable_kwargs = decorator_kwargs
             async_wrapper.__name__ = func.__name__
             return async_wrapper
        return func
    return decorator

# --- Test MCPClient Core Functionality ---

def test_mcpclient_init(mcp_client: MCPClient, mock_connector: MagicMock):
    """Test MCPClient initialization."""
    assert mcp_client._connector is mock_connector
    assert mcp_client._heartbeat_interval == 999
    assert not mcp_client._is_alive
    assert mcp_client._heartbeat_task is None
    assert isinstance(mcp_client.exit_stack, AsyncExitStack)
    assert isinstance(mcp_client._reconnect_lock, asyncio.Lock)

@pytest.mark.asyncio
async def test_mcpclient_connect_success(mcp_client: MCPClient, mock_connector: MagicMock, caplog):
    """Test successful connection and heartbeat start."""
    caplog.set_level(logging.INFO)
    mock_task_instance = AsyncMock(spec=asyncio.Task)
    mock_task_instance.cancel = MagicMock(return_value=True)
    mock_task_instance.__await__ = lambda: (yield)

    original_create_task = asyncio.create_task
    created_task_coro_name = None
    def create_task_wrapper(coro, *args, **kwargs):
        nonlocal created_task_coro_name
        if 'MCPClient._start_heartbeat' in getattr(coro, '__qualname__', ''):
             created_task_coro_name = coro.__qualname__
             return mock_task_instance
        else:
             return original_create_task(coro, *args, **kwargs)

    with patch('livekit_mcp_client.clients.mcpclient.asyncio.create_task', new=create_task_wrapper):
        await mcp_client.connect(timeout=5.0)

    mock_connector.connect.assert_awaited_once()
    assert mcp_client._is_alive is True
    assert created_task_coro_name == 'MCPClient._start_heartbeat'
    assert mcp_client._heartbeat_task is mock_task_instance
    assert "MCPClient connected successfully." in caplog.text

@pytest.mark.asyncio
async def test_mcpclient_connect_timeout(mcp_client: MCPClient, mock_connector: MagicMock):
    """Test connection timeout."""
    mock_connector.connect.side_effect = asyncio.TimeoutError("Connection timed out")

    with pytest.raises(MCPConnectionTimeoutError):
        await mcp_client.connect(timeout=0.1)

    mock_connector.connect.assert_awaited_once()
    assert mcp_client._is_alive is False
    assert mcp_client._heartbeat_task is None

@pytest.mark.asyncio
async def test_mcpclient_aclose(mcp_client: MCPClient, mock_connector: MagicMock):
    """Test closing the client."""
    mock_task_instance = AsyncMock(spec=asyncio.Task)
    mock_task_instance.cancel = MagicMock(return_value=True)
    mock_task_instance.__await__ = lambda: (yield asyncio.CancelledError())

    mcp_client._heartbeat_task = mock_task_instance
    mcp_client._is_alive = True

    await mcp_client.aclose()

    mock_task_instance.cancel.assert_called_once()
    mock_connector.disconnect.assert_awaited_once()
    assert mcp_client._is_alive is False
    assert mcp_client._heartbeat_task is None

@pytest.mark.asyncio
async def test_mcpclient_aclose_no_heartbeat(mcp_client: MCPClient, mock_connector: MagicMock):
    """Test closing when heartbeat wasn't started."""
    mcp_client._heartbeat_task = None
    mcp_client._is_alive = True

    await mcp_client.aclose()

    mock_connector.disconnect.assert_awaited_once()
    assert mcp_client._is_alive is False
    assert mcp_client._heartbeat_task is None

@pytest.mark.asyncio
async def test_mcpclient_is_alive(mcp_client: MCPClient, mock_connector: MagicMock):
    """Test is_alive delegates to connector."""
    mock_connector.is_alive.reset_mock()
    mock_connector.is_alive.return_value = True
    assert await mcp_client.is_alive() is True
    mock_connector.is_alive.assert_awaited_once()

    mock_connector.is_alive.reset_mock()
    mock_connector.is_alive.return_value = False
    assert await mcp_client.is_alive() is False
    mock_connector.is_alive.assert_awaited_once()

@pytest.mark.asyncio
async def test_mcpclient_heartbeat_reconnect_success(mcp_client: MCPClient, mock_connector: MagicMock, caplog, mock_shared_session: MagicMock):
    """Test heartbeat triggers successful reconnect via _check_liveness."""
    caplog.set_level(logging.INFO)
    mcp_client._heartbeat_interval = 0.05

    mock_connector.is_alive.reset_mock()
    mock_connector.disconnect.reset_mock()
    mock_connector.connect.reset_mock()
    mock_shared_session.send_ping.reset_mock()

    mock_connector.is_alive.side_effect = [True, True, True]
    mock_shared_session.send_ping.side_effect = [asyncio.TimeoutError("Ping timed out"), None, None]
    mock_connector.connect.side_effect = None
    mock_connector.connect.return_value = None

    mcp_client._is_alive = True
    heartbeat_task = asyncio.create_task(mcp_client._start_heartbeat())
    mcp_client._heartbeat_task = heartbeat_task

    tries = 0
    max_tries = 100
    while mock_connector.connect.call_count < 1 and tries < max_tries:
        await asyncio.sleep(0.02)
        tries += 1

    assert tries < max_tries, f"Connect was not called within timeout (count: {mock_connector.connect.call_count})"
    mock_connector.is_alive.assert_awaited()
    mock_shared_session.send_ping.assert_awaited()
    mock_connector.disconnect.assert_awaited_once()
    mock_connector.connect.assert_awaited_once()

    assert "Liveness check (ping) timed out" in caplog.text
    assert "Heartbeat detected dead connection, attempting reconnect..." in caplog.text
    assert any("Reconnected after 1 attempts" in record.message for record in caplog.records), "Reconnect success log missing"
    assert mcp_client._is_alive is True

    ping_call_count_after_reconnect = mock_shared_session.send_ping.call_count
    await asyncio.sleep(mcp_client._heartbeat_interval * 2.1)
    assert mock_shared_session.send_ping.call_count > ping_call_count_after_reconnect

    heartbeat_task.cancel()
    try:
        await asyncio.wait_for(heartbeat_task, timeout=0.2)
    except (asyncio.CancelledError, asyncio.TimeoutError):
        pass

@pytest.mark.asyncio
async def test_mcpclient_heartbeat_reconnect_failure(mcp_client: MCPClient, mock_connector: MagicMock, caplog, mock_shared_session: MagicMock):
    """Test heartbeat fails to reconnect after multiple attempts."""
    caplog.set_level(logging.WARNING)
    mcp_client._heartbeat_interval = 0.05
    try:
        sig = inspect.signature(mcp_client._reconnect)
        max_attempts = sig.parameters['max_attempts'].default
    except Exception:
        max_attempts = 5
    print(f"Using max_attempts = {max_attempts} for failure test.")

    mock_connector.is_alive.reset_mock()
    mock_connector.disconnect.reset_mock()
    mock_connector.connect.reset_mock()
    mock_shared_session.send_ping.reset_mock()

    mock_connector.is_alive.return_value = True
    mock_shared_session.send_ping.side_effect = ConnectionRefusedError("Ping always fails")
    mock_connector.connect.side_effect = MCPConnectionError("Failed to connect")

    mcp_client._is_alive = True
    heartbeat_task = asyncio.create_task(mcp_client._start_heartbeat())
    mcp_client._heartbeat_task = heartbeat_task

    try:
        total_backoff_time = sum(min(1.0 * (2 ** i), 30) for i in range(max_attempts))
        wait_timeout = total_backoff_time + max_attempts * mcp_client._heartbeat_interval + 5
        print(f"Waiting for heartbeat task to complete (timeout: {wait_timeout:.1f}s)")
        await asyncio.wait_for(heartbeat_task, timeout=wait_timeout)
    except asyncio.TimeoutError:
        pytest.fail(f"Heartbeat task did not complete within the expected timeout ({wait_timeout}s)")
    except Exception as e:
        print(f"Heartbeat task finished with exception: {e}")

    # --- Assertions ---
    assert mock_connector.disconnect.call_count >= max_attempts, f"Disconnect attempts ({mock_connector.disconnect.call_count}) did not reach {max_attempts}"
    assert mock_connector.connect.call_count >= max_attempts, f"Connect attempts ({mock_connector.connect.call_count}) did not reach {max_attempts}"

    assert f"Liveness check (ping) failed: Ping always fails" in caplog.text
    assert "Heartbeat detected dead connection, attempting reconnect..." in caplog.text
    assert f"Reconnect attempt {max_attempts} failed." in caplog.text
    assert "Heartbeat failed: Reconnect failed permanently. Stopping heartbeat." in caplog.text

    assert mcp_client._is_alive is False

# --- Test Tool Creation ---

TEST_SCHEMA_SIMPLE = {
    "type": "object", "properties": {
        "message": {"type": "string", "description": "The message to send"},
        "user_id": {"type": "integer"}
    }, "required": ["message"]
}
TEST_SCHEMA_COMPLEX = {
    "type": "object", "properties": {
        "items-list": {"type": "array", "items": {"type": "string"}},
        "optional_flag": {"type": ["boolean", "null"]},
        "nested obj": {"type": "object", "properties": {"a": {"type": "number"}}}
    }, "required": ["items-list"]
}

@pytest.mark.asyncio
@patch('livekit_mcp_client.clients.mcpclient.ai_callable', new=mock_ai_callable_impl)
async def test_create_mcp_tool_simple(mcp_client: MCPClient, mock_shared_session: MagicMock):
    """Test creating a simple tool function using builtins.compile."""
    reader = SchemaReader()
    tool_name = "SimpleTool"
    tool_desc = "A simple test tool"

    func = await MCPClient.create_mcp_tool(tool_name, tool_desc, TEST_SCHEMA_SIMPLE, reader)

    assert callable(func)
    assert hasattr(func, '_ai_callable_kwargs')
    assert func._ai_callable_kwargs['name'] == tool_name

    sig = inspect.signature(func)
    assert 'self' in sig.parameters
    assert 'message' in sig.parameters
    assert 'user_id' in sig.parameters
    assert sig.parameters['message'].annotation == str
    # --- Use CORRECT assertion for SchemaReader implementation ---
    assert sig.parameters['user_id'].annotation == int
    assert sig.parameters['user_id'].default is None
    # --- End correct assertion ---

    mock_shared_session.call_tool.return_value = MagicMock(content="Success")
    mock_shared_session.call_tool.reset_mock()
    mock_self = MagicMock(session=mock_shared_session)

    result = await func(mock_self, message="hello", user_id=123)
    mock_shared_session.call_tool.assert_awaited_once_with(tool_name, {"message": "hello", "user_id": 123})
    assert result == "Success"

@pytest.mark.asyncio
@patch('livekit_mcp_client.clients.mcpclient.ai_callable', new=mock_ai_callable_impl)
async def test_create_mcp_tool_complex(mcp_client: MCPClient, mock_shared_session: MagicMock):
    """Test creating a tool with complex types using builtins.compile."""
    reader = SchemaReader()
    tool_name = "Complex Tool-Name"
    tool_desc = "A complex test tool"

    func = await MCPClient.create_mcp_tool(tool_name, tool_desc, TEST_SCHEMA_COMPLEX, reader)

    assert callable(func)
    assert hasattr(func, '_ai_callable_kwargs')
    assert func._ai_callable_kwargs['name'] == tool_name

    mock_shared_session.call_tool.return_value = MagicMock(content="Complex Result")
    mock_shared_session.call_tool.reset_mock()
    mock_self = MagicMock(session=mock_shared_session)

    input_args = {"items_list": ["a", "b"], "optional_flag": True, "nested_obj": {"a": 1.2}}
    expected_call_kwargs = {"items-list": ["a", "b"], "optional_flag": True, "nested obj": {"a": 1.2}}

    result = await func(mock_self, **input_args)
    mock_shared_session.call_tool.assert_awaited_once_with(tool_name, expected_call_kwargs)
    assert result == "Complex Result"

@pytest.mark.asyncio
async def test_create_mcp_tool_compilation_error(mcp_client: MCPClient, caplog):
    """Test handling of compile errors using builtins.compile."""
    reader = SchemaReader()
    tool_name = "BadSyntaxTool"
    tool_desc = "This tool has bad syntax"
    bad_schema = {"type": "object", "properties": {"ok_name": {"type": "string"}}}

    with patch('builtins.compile') as mock_compile:
        compile_exception = SyntaxError("Invalid Python generated")
        mock_compile.side_effect = compile_exception

        with pytest.raises(ValueError, match=f"Tool creation failed for {tool_name}") as exc_info:
            await MCPClient.create_mcp_tool(tool_name, tool_desc, bad_schema, reader)

        assert f"Failed to create tool {tool_name}: {compile_exception}" in caplog.text
        assert exc_info.value.__cause__ is compile_exception

# --- Test FunctionContext Creation ---

@pytest.mark.asyncio
@patch('livekit_mcp_client.clients.mcpclient.ai_callable', new=mock_ai_callable_impl)
async def test_create_function_context_success(mcp_client: MCPClient, mock_connector: MagicMock, mock_shared_session: MagicMock, caplog):
    """Test creating FunctionContext with multiple tools."""
    mcpclient_logger = logging.getLogger('livekit_mcp_client.clients.mcpclient')
    original_level = mcpclient_logger.level
    mcpclient_logger.setLevel(logging.DEBUG)
    caplog.set_level(logging.DEBUG)

    reader = SchemaReader()
    try:
        tool1_mock = MagicMock(description="First tool", inputSchema=TEST_SCHEMA_SIMPLE)
        tool1_mock.name = "ToolOne"
        tool2_mock = MagicMock(description="Second tool", inputSchema=TEST_SCHEMA_COMPLEX)
        tool2_mock.name = "Tool Two"
        mock_list_response = MagicMock(tools=[tool1_mock, tool2_mock])
        mock_shared_session.list_tools.return_value = mock_list_response
        mock_shared_session.call_tool.return_value = MagicMock(content="Tool Result")
        mock_shared_session.list_tools.reset_mock()
        mock_shared_session.call_tool.reset_mock()

        # --- Action ---
        fcn_ctx = await mcp_client.create_function_context(reader)

        # --- Assertions ---
        mock_shared_session.list_tools.assert_awaited_once()
        assert "Fetching tools from MCP Server" in caplog.text
        assert f"Successfully registered tool: {tool1_mock.name}" in caplog.text
        assert f"Successfully registered tool: {tool2_mock.name}" in caplog.text
        assert "FunctionContext built with 2/2 tools" in caplog.text

        assert isinstance(fcn_ctx, FunctionContext)
        assert hasattr(fcn_ctx, "session")
        assert fcn_ctx.session is mock_shared_session

        sanitized_name1 = utils.sanitize_name(tool1_mock.name)
        sanitized_name2 = utils.sanitize_name(tool2_mock.name)
        assert hasattr(fcn_ctx, sanitized_name1)
        assert hasattr(fcn_ctx, sanitized_name2)
        assert callable(getattr(fcn_ctx, sanitized_name1))
        assert callable(getattr(fcn_ctx, sanitized_name2))

        assert getattr(fcn_ctx, sanitized_name1)._ai_callable_kwargs == {'name': tool1_mock.name, 'description': tool1_mock.description}
        assert getattr(fcn_ctx, sanitized_name2)._ai_callable_kwargs == {'name': tool2_mock.name, 'description': tool2_mock.description}

        await getattr(fcn_ctx, sanitized_name1)(message="test")
        mock_shared_session.call_tool.assert_awaited_with(tool1_mock.name, {"message": "test"})

    finally:
        mcpclient_logger.setLevel(original_level)


@pytest.mark.asyncio
@patch('livekit_mcp_client.clients.mcpclient.ai_callable', new=mock_ai_callable_impl)
async def test_create_function_context_with_failures(mcp_client: MCPClient, mock_connector: MagicMock, mock_shared_session: MagicMock, caplog):
    """Test creating FunctionContext when some tools fail to build."""
    caplog.set_level(logging.INFO)
    reader = SchemaReader()

    good_tool_mock = MagicMock(description="Works fine", inputSchema=TEST_SCHEMA_SIMPLE)
    good_tool_mock.name = "GoodTool"
    bad_tool_mock = MagicMock(description="Will fail build", inputSchema={"type": "object", "properties": {}}, name="BadTool")
    bad_tool_mock.name = "BadTool"
    ugly_tool_mock = MagicMock(description="Invalid schema", inputSchema={"type": "object"}, name="UglyTool")
    ugly_tool_mock.name = "UglyTool"
    mock_list_response = MagicMock(tools=[good_tool_mock, bad_tool_mock, ugly_tool_mock])
    mock_shared_session.list_tools.return_value = mock_list_response
    mock_shared_session.list_tools.reset_mock()

    original_create = MCPClient.create_mcp_tool
    async def mock_create_tool_with_failure(self, tool_name: str, tool_desc: str, input_schema: dict, *, reader: SchemaReader):
        if tool_name == bad_tool_mock.name:
            raise ValueError("Intentional failure for BadTool")
        return await original_create(tool_name, tool_desc, input_schema, reader=reader)

    with patch.object(MCPClient, 'create_mcp_tool', new=mock_create_tool_with_failure):
         fcn_ctx = await mcp_client.create_function_context(reader)

    # --- Assertions ---
    mock_shared_session.list_tools.assert_awaited_once()

    assert f"Skipping tool '{bad_tool_mock.name}' due to error: Intentional failure for BadTool" in caplog.text
    assert f"Skipping tool '{ugly_tool_mock.name}'" in caplog.text
    assert "Schema must have 'properties' as a dict" in caplog.text

    failed_names = sorted([bad_tool_mock.name, ugly_tool_mock.name])
    assert f"Skipped {len(failed_names)} tools due to errors: {', '.join(failed_names)}" in caplog.text
    assert "FunctionContext built with 1/3 tools" in caplog.text

    assert isinstance(fcn_ctx, FunctionContext)
    sanitized_good_name = utils.sanitize_name(good_tool_mock.name)
    sanitized_bad_name = utils.sanitize_name(bad_tool_mock.name)
    sanitized_ugly_name = utils.sanitize_name(ugly_tool_mock.name)

    assert hasattr(fcn_ctx, sanitized_good_name)
    assert not hasattr(fcn_ctx, sanitized_bad_name)
    assert not hasattr(fcn_ctx, sanitized_ugly_name)
    assert getattr(fcn_ctx, sanitized_good_name)._ai_callable_kwargs['name'] == good_tool_mock.name