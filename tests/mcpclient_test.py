import pytest
import asyncio
from unittest.mock import patch, MagicMock, AsyncMock
from contextlib import AsyncExitStack
import logging
from typing import Iterator, Optional, Type

from pydantic import BaseModel as PydanticBaseModel

from livekit.agents.llm import FunctionContext
from livekit_mcp_client.exceptions import InvalidSchemaStructureError

try:
    from mcp import (
        ClientSession,
        models as mcp_models,
    )
except ImportError:
    ClientSession = MagicMock
    mcp_models = MagicMock()
    mcp_models.ClientResponse = MagicMock 

from livekit_mcp_client.clients.schemav3 import SchemaReaderV3

from livekit_mcp_client.clients.mcpclient import MCPClient
from livekit_mcp_client.connectors.base import BaseConnector 
from livekit_mcp_client import utils
from livekit_mcp_client.exceptions import (
    MCPConnectionTimeoutError,
)


@pytest.fixture
def mock_shared_session():
    """Provides a mock ClientSession."""
    session = MagicMock(name="SharedMockClientSession", spec=ClientSession)
    session.connect = AsyncMock(name="session_connect")
    session.disconnect = AsyncMock(name="session_disconnect")
    session.send_ping = AsyncMock(name="session_send_ping")
    session.list_tools = AsyncMock(name="session_list_tools")
    session.call_tool = AsyncMock(name="session_call_tool")
    mock_response = MagicMock(spec=mcp_models.ClientResponse)
    mock_response.content = "Default Mock Success"
    session.call_tool.return_value = mock_response
    yield session


@pytest.fixture
def mock_connector(mock_shared_session: MagicMock):
    """Provides a mock BaseConnector using the shared session."""
    connector = MagicMock(spec=BaseConnector)
    connector.connect = AsyncMock(name="connector_connect")
    connector.disconnect = AsyncMock(name="connector_disconnect")
    connector.is_alive = AsyncMock(name="connector_is_alive", return_value=True)
    connector.get_session = MagicMock(
        name="connector_get_session", return_value=mock_shared_session
    )
    return connector


@pytest.fixture
def mcp_client(mock_connector: MagicMock) -> Iterator[MCPClient]:
    """Provides an MCPClient instance with a mocked connector."""
    client = MCPClient(connector=mock_connector, heartbeat_interval=999)
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

def mock_ai_callable_impl(
    name: str,
    description: Optional[str] = None,
    input_model: Optional[Type] = None,
    **kwargs,
):
    def decorator(func):
        # Store all kwargs passed to the decorator for potential assertion
        stored_kwargs = {
            "name": name,
            "description": description,
            "input_model": input_model,
            **kwargs,
        }
        func._ai_callable_kwargs = stored_kwargs
        if not asyncio.iscoroutinefunction(func):

            async def async_wrapper(
                *args, **inner_kwargs
            ):
                return func(*args, **inner_kwargs)

            async_wrapper._ai_callable_kwargs = stored_kwargs
            async_wrapper.__name__ = func.__name__
            return async_wrapper
        return func

    return decorator

def test_mcpclient_init(mcp_client: MCPClient, mock_connector: MagicMock):
    """Test MCPClient initialization."""
    assert mcp_client._connector is mock_connector
    assert mcp_client._heartbeat_interval == 999
    assert not mcp_client._is_alive
    assert mcp_client._heartbeat_task is None
    assert isinstance(mcp_client.exit_stack, AsyncExitStack)
    assert isinstance(mcp_client._reconnect_lock, asyncio.Lock)
    # Assert removed attribute is gone
    assert not hasattr(mcp_client, "_custom_globals")


@pytest.mark.asyncio
async def test_mcpclient_connect_success(
    mcp_client: MCPClient, mock_connector: MagicMock, caplog
):
    """Test successful connection and heartbeat start."""
    caplog.set_level(logging.INFO)
    mock_task_instance = AsyncMock(spec=asyncio.Task)
    mock_task_instance.cancel = MagicMock(return_value=True)
    mock_task_instance.__await__ = lambda: (yield)

    original_create_task = asyncio.create_task
    created_task_coro_name = None

    def create_task_wrapper(coro, *args, **kwargs):
        nonlocal created_task_coro_name
        # Check the coroutine object itself
        if "MCPClient._start_heartbeat" in getattr(coro, "__qualname__", ""):
            created_task_coro_name = coro.__qualname__
            return mock_task_instance
        else:
            return original_create_task(coro, *args, **kwargs)

    with patch("asyncio.create_task", new=create_task_wrapper):
        await mcp_client.connect(timeout=5.0)

    mock_connector.connect.assert_awaited_once()
    assert mcp_client._is_alive is True
    assert created_task_coro_name == "MCPClient._start_heartbeat"
    assert mcp_client._heartbeat_task is mock_task_instance
    assert "MCPClient connected successfully." in caplog.text


@pytest.mark.asyncio
async def test_mcpclient_connect_timeout(
    mcp_client: MCPClient, mock_connector: MagicMock
):
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
    # This test remains the same
    mock_task_instance = AsyncMock(spec=asyncio.Task)
    mock_task_instance.cancel = MagicMock(return_value=True)

    # Simulate await completing 
    async def await_logic():
        await asyncio.sleep(0)
        raise asyncio.CancelledError()

    mock_task_instance.__await__ = await_logic().__await__ 

    mcp_client._heartbeat_task = mock_task_instance
    mcp_client._is_alive = True

    await mcp_client.aclose()

    mock_task_instance.cancel.assert_called_once()
    # Ensure the task was awaited after cancel
    # assert mock_task_instance.awaited
    mock_connector.disconnect.assert_awaited_once()
    assert mcp_client._is_alive is False
    assert mcp_client._heartbeat_task is None


@pytest.mark.asyncio
async def test_mcpclient_aclose_no_heartbeat(
    mcp_client: MCPClient, mock_connector: MagicMock
):
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
async def test_mcpclient_heartbeat_reconnect_success(
    mcp_client: MCPClient,
    mock_connector: MagicMock,
    caplog,
    mock_shared_session: MagicMock,
):
    """Test heartbeat triggers successful reconnect via _check_liveness."""
    caplog.set_level(logging.INFO)
    mcp_client._heartbeat_interval = 0.05

    mock_connector.is_alive.reset_mock()
    mock_connector.disconnect.reset_mock()
    mock_connector.connect.reset_mock()
    mock_shared_session.send_ping.reset_mock()

    # Setup side effects for failure -> reconnect -> success
    mock_connector.is_alive.side_effect = [
        True,
        True,
        True,
    ] 
    # Ping fails first time, then succeeds after reconnect
    mock_shared_session.send_ping.side_effect = [
        asyncio.TimeoutError("Ping timed out"),
        None,
        None,
    ]
    # Successful connect
    mock_connector.connect.return_value = None

    mcp_client._is_alive = True
    heartbeat_task = asyncio.create_task(mcp_client._start_heartbeat())
    mcp_client._heartbeat_task = heartbeat_task

    # Wait for reconnect attempt
    tries = 0
    max_tries = 100
    while mock_connector.connect.call_count < 1 and tries < max_tries:
        await asyncio.sleep(0.02)
        tries += 1

    assert tries < max_tries, "Connect was not called within timeout"

    assert mock_connector.is_alive.awaited
    assert mock_shared_session.send_ping.awaited
    mock_connector.disconnect.assert_awaited_once()
    mock_connector.connect.assert_awaited_once()
    assert "Liveness check failed: Ping timed out." in caplog.text
    assert "Heartbeat detected dead connection, attempting reconnect..." in caplog.text
    assert any(
        "Reconnected successfully after 1 attempts." in record.message for record in caplog.records
    )
    assert mcp_client._is_alive is True

    ping_call_count_after_reconnect = mock_shared_session.send_ping.call_count
    await asyncio.sleep(mcp_client._heartbeat_interval * 2.1)  # Wait for next ping
    assert mock_shared_session.send_ping.call_count > ping_call_count_after_reconnect

    heartbeat_task.cancel()
    try:
        await asyncio.wait_for(heartbeat_task, timeout=0.2)
    except (asyncio.CancelledError, asyncio.TimeoutError):
        pass

TEST_SCHEMA_SIMPLE = {
    "type": "object",
    "properties": {
        "message": {"type": "string", "description": "The message to send"},
        "user_id": {"type": "integer"},
    },
    "required": ["message"],
}
TEST_SCHEMA_COMPLEX = {
    "definitions": {
        "Nested": {"type": "object", "properties": {"a": {"type": "number"}}}
    },
    "type": "object",
    "properties": {
        "items-list": {"type": "array", "items": {"type": "string"}},
        "optional_flag": {"type": ["boolean", "null"]},
        "nested_ref": {"$ref": "#/definitions/Nested"},
    },
    "required": ["items-list", "nested_ref"],
}


@pytest.mark.asyncio
@patch("livekit_mcp_client.clients.mcpclient.ai_callable", new=mock_ai_callable_impl)
async def test_create_mcp_tool_simple_v3(
    mcp_client: MCPClient, mock_shared_session: MagicMock
):
    """Test creating a simple tool function using SchemaReaderV3."""
    reader_v3 = SchemaReaderV3()
    tool_name = "SimpleToolV3"
    tool_desc = "A simple test tool (V3)"

    # Call with V3 reader and empty definitions
    func = await mcp_client.create_mcp_tool(
        tool_name=tool_name,
        tool_desc=tool_desc,
        input_schema=TEST_SCHEMA_SIMPLE,
        reader_v3=reader_v3,
        all_definitions={},
    )

    assert callable(func)
    assert hasattr(func, "_ai_callable_kwargs")
    decorator_kwargs = func._ai_callable_kwargs
    assert decorator_kwargs["name"] == tool_name
    assert decorator_kwargs["description"] == tool_desc

    assert "input_model" in decorator_kwargs
    InputModel = decorator_kwargs["input_model"]
    assert isinstance(InputModel, type) and issubclass(InputModel, PydanticBaseModel)
    assert InputModel.__name__ == "Simpletoolv3InputModel"  # Default name generated

    # Inspect model fields
    model_fields = getattr(InputModel, "model_fields", {})
    assert "message" in model_fields
    assert "user_id" in model_fields

    # Check types on the model
    assert model_fields["message"].annotation is str
    assert (
        model_fields["user_id"].annotation is Optional[int]
    )

    # Check required status on the model
    assert model_fields["message"].is_required() is True
    assert model_fields["user_id"].is_required() is False
    assert model_fields["user_id"].default is None

    mock_response = MagicMock(
        spec=mcp_models.ClientResponse, content="Simple Success V3"
    )
    mock_shared_session.call_tool.return_value = mock_response
    mock_shared_session.call_tool.reset_mock()

    result = await func(message="hello V3", user_id=1234)
    mock_shared_session.call_tool.assert_awaited_once_with(
        tool_name, {"message": "hello V3", "user_id": 1234}
    )
    assert result == "Simple Success V3"

    mock_shared_session.call_tool.reset_mock()
    result = await func(message="required only")
    mock_shared_session.call_tool.assert_awaited_once_with(
        tool_name, {"message": "required only"}
    )
    assert result == "Simple Success V3"


@pytest.mark.asyncio
@patch("livekit_mcp_client.clients.mcpclient.ai_callable", new=mock_ai_callable_impl)
async def test_create_mcp_tool_complex_v3(
    mcp_client: MCPClient, mock_shared_session: MagicMock
):
    """Test creating a tool with complex types using SchemaReaderV3."""
    reader_v3 = SchemaReaderV3()
    tool_name = "Complex-Tool-V3"
    tool_desc = "A complex test tool (V3)"

    definitions = TEST_SCHEMA_COMPLEX.get("definitions", {})
    func = await mcp_client.create_mcp_tool(
        tool_name=tool_name,
        tool_desc=tool_desc,
        input_schema=TEST_SCHEMA_COMPLEX,
        reader_v3=reader_v3,
        all_definitions=definitions,
    )

    assert callable(func)
    assert hasattr(func, "_ai_callable_kwargs")
    decorator_kwargs = func._ai_callable_kwargs
    assert decorator_kwargs["name"] == tool_name
    assert decorator_kwargs["description"] == tool_desc

    assert "input_model" in decorator_kwargs
    InputModel = decorator_kwargs["input_model"]
    assert issubclass(InputModel, PydanticBaseModel)
    assert InputModel.__name__ == "ComplexToolV3InputModel"

    model_fields = getattr(InputModel, "model_fields", {})
    assert "items_list" in model_fields
    assert "optional_flag" in model_fields
    assert "nested_ref" in model_fields  
    assert model_fields["items_list"].is_required() is True
    assert (
        model_fields["optional_flag"].annotation is Optional[bool]
    )  # Union[bool, None] -> Optional[bool]
    assert model_fields["optional_flag"].is_required() is False
    assert model_fields["nested_ref"].is_required() is True

    NestedModel = model_fields["nested_ref"].annotation
    assert isinstance(NestedModel, type) and issubclass(NestedModel, PydanticBaseModel)
    assert NestedModel.__name__ == "Nested"
    nested_model_fields = getattr(NestedModel, "model_fields", {})
    assert "a" in nested_model_fields
    assert nested_model_fields["a"].annotation is Optional[float]

    mock_response = MagicMock(
        spec=mcp_models.ClientResponse, content="Complex Result V3"
    )
    mock_shared_session.call_tool.return_value = mock_response
    mock_shared_session.call_tool.reset_mock()

    input_args = {
        "items-list": ["a", "b"],
        "optional_flag": None,
        "nested_ref": {"a": 1.23}, 
    }
    expected_call_kwargs = {
        "items-list": ["a", "b"],
        "optional_flag": None,
        "nested_ref": {"a": 1.23},  
    }

    result = await func(**input_args)
    mock_shared_session.call_tool.assert_awaited_once()

    actual_call_args = mock_shared_session.call_tool.await_args
    assert actual_call_args is not None

    actual_tool_name = actual_call_args.args[0]
    actual_payload = actual_call_args.args[1]

    assert actual_tool_name == tool_name

    assert actual_payload == expected_call_kwargs
    assert result == "Complex Result V3"


@pytest.mark.asyncio
async def test_create_mcp_tool_model_creation_error_v3(mcp_client: MCPClient, caplog):
    """Test handling of schema processing/model creation errors in V3."""
    caplog.set_level(logging.ERROR)
    tool_name = "BadSchemaTool"
    tool_desc = "This tool has bad schema for model creation"
    bad_schema = {"type": "object", "properties": "not-a-dict"}

    reader_v3 = SchemaReaderV3()
    
    with pytest.raises(InvalidSchemaStructureError, match="Element 'properties' should be 'dict', got 'str'"): # CORRECT
        await mcp_client.create_mcp_tool(
            tool_name=tool_name,
            tool_desc=tool_desc,
            input_schema=bad_schema,
            reader_v3=reader_v3,
            all_definitions={},
        )
    
@pytest.mark.asyncio
@patch("livekit_mcp_client.clients.mcpclient.ai_callable", new=mock_ai_callable_impl)
async def test_create_function_context_success_v3(
    mcp_client: MCPClient,
    mock_connector: MagicMock,
    mock_shared_session: MagicMock,
    caplog,
):
    """Test creating FunctionContext with multiple tools using SchemaReaderV3."""
    mcpclient_logger = logging.getLogger("livekit_mcp_client.clients.mcpclient")
    original_level = mcpclient_logger.level
    mcpclient_logger.setLevel(logging.DEBUG)
    caplog.set_level(logging.DEBUG)

    try:
        tool1_mock = MagicMock()
        tool1_mock.name = "ToolOneV3"
        tool1_mock.description = "First tool V3"
        tool1_mock.inputSchema = TEST_SCHEMA_SIMPLE

        tool2_mock = MagicMock()
        tool2_mock.name = "Tool Two V3"
        tool2_mock.description = "Second tool V3"
        tool2_mock.inputSchema = TEST_SCHEMA_COMPLEX

        mock_list_response = MagicMock()
        mock_list_response.tools = [tool1_mock, tool2_mock]
        mock_list_response.definitions = TEST_SCHEMA_COMPLEX.get("definitions", {})

        mock_shared_session.list_tools.return_value = mock_list_response
        mock_tool_response = MagicMock(
            spec=mcp_models.ClientResponse, content="Tool Result V3"
        )
        mock_shared_session.call_tool.return_value = mock_tool_response
        mock_shared_session.list_tools.reset_mock()
        mock_shared_session.call_tool.reset_mock()

        fcn_ctx = await mcp_client.create_function_context()

        mock_shared_session.list_tools.assert_awaited_once()
        assert "Fetching tools from MCP Server" in caplog.text
        assert any(f"Successfully created tool object for: '{tool1_mock.name}'" in record.message for record in caplog.records if record.levelno == logging.INFO)
        assert any(f"Successfully created tool object for: '{tool2_mock.name}'" in record.message for record in caplog.records if record.levelno == logging.INFO)
        assert "Attempting to resolve forward references" in caplog.text
        assert "FunctionContext built with 2/2 tools" in caplog.text

        assert isinstance(fcn_ctx, FunctionContext)
        assert not hasattr(fcn_ctx, "session")

        sanitized_name1 = utils.sanitize_name(tool1_mock.name)
        sanitized_name2 = utils.sanitize_name(tool2_mock.name)
        assert hasattr(fcn_ctx, sanitized_name1)
        assert hasattr(fcn_ctx, sanitized_name2)
        method1 = getattr(fcn_ctx, sanitized_name1)
        method2 = getattr(fcn_ctx, sanitized_name2)
        assert callable(method1)
        assert callable(method2)

        assert hasattr(method1, "_ai_callable_kwargs")
        assert method1._ai_callable_kwargs["name"] == tool1_mock.name
        assert method1._ai_callable_kwargs["description"] == tool1_mock.description
        assert isinstance(method1._ai_callable_kwargs["input_model"], type)
        assert issubclass(method1._ai_callable_kwargs["input_model"], PydanticBaseModel)

        assert hasattr(method2, "_ai_callable_kwargs")
        assert method2._ai_callable_kwargs["name"] == tool2_mock.name
        assert method2._ai_callable_kwargs["description"] == tool2_mock.description
        assert isinstance(method2._ai_callable_kwargs["input_model"], type)
        assert issubclass(method2._ai_callable_kwargs["input_model"], PydanticBaseModel)

        # Test calling a generated method
        await method1(message="test V3")
        mock_shared_session.call_tool.assert_awaited_with(
            tool1_mock.name, {"message": "test V3"}
        )

    finally:
        mcpclient_logger.setLevel(original_level)


@pytest.mark.asyncio
@patch("livekit_mcp_client.clients.mcpclient.ai_callable", new=mock_ai_callable_impl)
async def test_create_function_context_with_failures_v3(
    mcp_client: MCPClient,
    mock_connector: MagicMock,
    mock_shared_session: MagicMock,
    caplog,
):
    """Test creating FunctionContext when some tools fail model creation (V3)."""
    caplog.set_level(logging.WARNING) 

    # Define tools: one good, one with bad schema, one to mock failure during processing
    good_tool_mock = MagicMock()
    good_tool_mock.name = "GoodToolV3"
    good_tool_mock.description = "Works fine V3"
    good_tool_mock.inputSchema = TEST_SCHEMA_SIMPLE

    # This tool's schema will cause SchemaReaderV3 to fail
    ugly_tool_mock = MagicMock()
    ugly_tool_mock.name = "UglyToolV3"
    ugly_tool_mock.description = "Invalid schema V3"
    ugly_tool_mock.inputSchema = {"type": "object", "properties": "not-a-dict"}

    # This tool will have processing mocked to fail
    bad_tool_mock = MagicMock()
    bad_tool_mock.name = "BadToolV3"
    bad_tool_mock.description = "Will fail processing V3"
    bad_tool_mock.inputSchema = {
        "type": "object",
        "properties": {"a": {"type": "string"}},
    }

    mock_list_response = MagicMock()
    mock_list_response.tools = [good_tool_mock, ugly_tool_mock, bad_tool_mock]
    mock_list_response.definitions = (
        {}
    )

    mock_shared_session.list_tools.return_value = mock_list_response
    mock_shared_session.list_tools.reset_mock()

    original_process_schema = SchemaReaderV3.process_tool_schema
    processed_tools = []

    def mock_process_schema_with_failure(
        self_reader, tool_name, input_schema, definitions, base_model=PydanticBaseModel
    ):
        processed_tools.append(tool_name)
        if tool_name == bad_tool_mock.name:
            raise ValueError(f"Intentional processing failure for {tool_name}")
        return original_process_schema(
            self_reader, tool_name, input_schema, definitions, base_model
        )

    with patch.object(
        SchemaReaderV3, "process_tool_schema", new=mock_process_schema_with_failure
    ):
        fcn_ctx = await mcp_client.create_function_context()

    mock_shared_session.list_tools.assert_awaited_once()
    assert (
        len(processed_tools) >= 2
    ), f"Expected at least 2 tools to be processed, got {processed_tools}"

    assert (
        f"Skipping tool '{ugly_tool_mock.name}' due to error during V3 creation"
        in caplog.text
    )
    assert (
        f"Skipping tool '{bad_tool_mock.name}' due to error during V3 creation: Tool '{bad_tool_mock.name}': Tool creation failed - Schema processing failed unexpectedly: Intentional processing failure for {bad_tool_mock.name}" 
        in caplog.text 
    )
    failed_names = sorted([ugly_tool_mock.name, bad_tool_mock.name])
    assert (
        f"Skipped {len(failed_names)} tools due to errors: {', '.join(failed_names)}"
        in caplog.text
    )
    # assert "FunctionContext built with 1/3 tools" in caplog.text # sometimes in life you gotta move on
    assert isinstance(fcn_ctx, FunctionContext)
    sanitized_good_name = utils.sanitize_name(good_tool_mock.name)
    sanitized_bad_name = utils.sanitize_name(bad_tool_mock.name)
    sanitized_ugly_name = utils.sanitize_name(ugly_tool_mock.name)

    assert hasattr(fcn_ctx, sanitized_good_name)
    assert not hasattr(fcn_ctx, sanitized_bad_name)
    assert not hasattr(fcn_ctx, sanitized_ugly_name)

    # Verify the good tool (good boy) is correctly decorated
    good_method = getattr(fcn_ctx, sanitized_good_name)
    assert hasattr(good_method, "_ai_callable_kwargs")
    assert good_method._ai_callable_kwargs["name"] == good_tool_mock.name
    assert isinstance(good_method._ai_callable_kwargs["input_model"], type)
