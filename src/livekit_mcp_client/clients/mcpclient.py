# --- START OF FILE mcpclient.py ---

from typing import Optional, Any, Callable, Dict, Type, List
from contextlib import AsyncExitStack
import logging
import asyncio
from dotenv import load_dotenv

from livekit_mcp_client.clients.schemav3 import SchemaReaderV3
from livekit.agents.llm import ai_callable, FunctionContext
from mcp import ClientSession

from livekit_mcp_client.exceptions import (
    MCPConnectionTimeoutError,
    MCPConnectionError,
    MissingConnectionException,
    CorruptConnectionException,
    MCPReconnectError,
    SchemaError,
    ToolCreationError,
    ToolInputValidationError,
    ToolExecutionError,
)
from livekit_mcp_client.connectors.base import BaseConnector
from livekit_mcp_client.utils import sanitize_name
from pydantic import BaseModel as PydanticBaseModel

load_dotenv()

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class MCPClient:
    """Client for interacting with an MCP server via a connector.

    Manages connection, heartbeat, reconnection, and dynamic tool generation.
    """

    def __init__(self, connector: BaseConnector, heartbeat_interval: int = 30):
        """Initializes the MCP client.

        Args:
            connector: The connector instance handling communication.
            heartbeat_interval: Interval (seconds) for liveness checks (0=disabled).
        """
        self.exit_stack = AsyncExitStack()
        self._connector = connector
        self._is_alive = False
        self._heartbeat_task: Optional[asyncio.Task] = None
        self._heartbeat_interval = heartbeat_interval
        self._reconnect_lock = asyncio.Lock()

    def __del__(self):
        """Warns if client wasn't explicitly closed via aclose()."""
        if self._is_alive:
            logger.warning(
                "MCPClient instance garbage collected while still marked as alive. Consider using aclose()."
            )

    async def __aenter__(self):
        """Async context manager entry point."""
        await self.exit_stack.__aenter__()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit point, ensures cleanup via aclose()."""
        logger.debug("Exiting async context, calling aclose().")
        await self.aclose()
        await self.exit_stack.__aexit__(exc_type, exc_val, exc_tb)

    def get_session(self) -> ClientSession:
        """Retrieves the MCP ClientSession from the connector.

        Raises:
            MissingConnectionException: If connector is missing.
            CorruptConnectionException: If connector fails to provide a session.
        """
        if self._connector is None:
            logger.error("Connector is None when getting session.")
            raise MissingConnectionException()
        session = self._connector.get_session()
        if session is None:
            logger.error("Connector returned None session.")
            raise CorruptConnectionException()
        return session

    async def _check_liveness(self) -> bool:
        """Checks connection status via connector and MCP ping."""
        if not await self._connector.is_alive():
            logger.debug("Liveness check failed: Connector reported not alive.")
            return False
        try:
            session = self.get_session()
            await asyncio.wait_for(session.send_ping(), timeout=5.0)
            logger.debug("Liveness check passed (ping successful).")
            return True
        except (MissingConnectionException, CorruptConnectionException):
            logger.warning("Cannot check liveness, connection state issue.")
            return False
        except asyncio.TimeoutError:
            logger.warning("Liveness check failed: Ping timed out.")
            return False
        except Exception as e:
            logger.warning(f"Liveness check failed: Ping raised exception: {e}")
            return False

    async def _start_heartbeat(self) -> None:
        """Background task checking connection and triggering reconnects."""
        logger.info("Heartbeat task started.")
        while True:
            await asyncio.sleep(self._heartbeat_interval)
            logger.debug("Heartbeat interval elapsed, checking liveness...")
            try:
                if not await self._check_liveness():
                    logger.warning(
                        "Heartbeat detected dead connection, attempting reconnect..."
                    )
                    await self._reconnect()
                else:
                    logger.debug("Heartbeat check passed.")
            except MCPReconnectError:
                logger.error("Heartbeat stopping: Reconnect failed permanently.")
                self._is_alive = False
                break
            except asyncio.CancelledError:
                logger.info("Heartbeat task cancelled.")
                break
            except Exception as e:
                logger.error(f"Unexpected error in heartbeat loop: {e}", exc_info=True)
                logger.warning("Attempting reconnect after unexpected error...")
                try:
                    await self._reconnect()
                except MCPReconnectError:
                    logger.error(
                        "Heartbeat stopping: Reconnect failed after unexpected error."
                    )
                    self._is_alive = False
                    break

    async def _reconnect(self, max_attempts: int = 5, base_delay: float = 1.0) -> None:
        """Attempts reconnection with exponential backoff.

        Raises:
            MCPReconnectError: If all attempts fail.
        """
        logger.info(f"Attempting reconnection (max {max_attempts} attempts)...")
        for attempt in range(max_attempts):
            try:
                async with self._reconnect_lock:
                    logger.debug(
                        f"Reconnect attempt {attempt + 1}: Disconnecting/Connecting..."
                    )
                    await self._connector.disconnect()
                    await self._connector.connect()
                self._is_alive = True
                logger.info(f"Reconnected successfully after {attempt + 1} attempts.")
                return
            except Exception as e:
                delay = min(base_delay * (2**attempt), 30)
                logger.warning(
                    f"Reconnect attempt {attempt + 1} failed: {e}. Retrying in {delay:.1f}s..."
                )
                await asyncio.sleep(delay)
        logger.error("Reconnection failed after maximum attempts.")
        raise MCPReconnectError(f"Failed to reconnect after {max_attempts} attempts")

    async def aclose(self) -> None:
        """Closes the connection and cleans up resources (heartbeat task)."""
        logger.info("Closing MCPClient...")
        if self._heartbeat_task:
            logger.debug("Cancelling heartbeat task...")
            self._heartbeat_task.cancel()
            try:
                await self._heartbeat_task
            except asyncio.CancelledError:
                logger.debug("Heartbeat task successfully cancelled.")
            except Exception as e:
                logger.error(
                    f"Error awaiting cancelled heartbeat task: {e}", exc_info=True
                )
            finally:
                self._heartbeat_task = None
        # Disconnect connector
        try:
            if self._connector:
                logger.debug("Disconnecting connector...")
                await self._connector.disconnect()
                logger.debug("Connector disconnected.")
        except Exception as e:
            logger.error(f"Error during connector disconnect: {e}", exc_info=True)
        finally:
            self._is_alive = False
            logger.info("MCPClient closed.")

    async def is_alive(self) -> bool:
        """Checks if the client connection is considered active (delegates to connector)."""
        if not self._connector:
            return False
        return await self._connector.is_alive()

    async def connect(self, timeout: float = 10.0) -> None:
        """Connects to the MCP server and starts the heartbeat.

        Args:
            timeout: Connection attempt timeout in seconds.

        Raises:
            MCPConnectionTimeoutError: If connection times out.
            MCPConnectionError: For other connection failures.
        """
        logger.info(f"Attempting to connect (timeout: {timeout}s)...")
        try:
            await asyncio.wait_for(self._connector.connect(), timeout)
            self._is_alive = True
            # Start heartbeat if enabled and not already running
            if self._heartbeat_interval > 0 and self._heartbeat_task is None:
                logger.debug("Creating heartbeat task...")
                self._heartbeat_task = asyncio.create_task(self._start_heartbeat())
            logger.info("MCPClient connected successfully.")
        except asyncio.TimeoutError as e:
            self._is_alive = False
            logger.error(f"Connection timed out after {timeout}s.")
            raise MCPConnectionTimeoutError(timeout) from e
        except Exception as e:
            self._is_alive = False
            logger.error(f"MCPClient connection failed: {e}", exc_info=True)
            raise MCPConnectionError(f"Connection failed: {e}") from e

    def _create_tool_wrapper_func(
        self, tool_name: str, InputModel: Type[PydanticBaseModel]
    ) -> Callable:
        """Internal factory creating an async wrapper function for an MCP tool call."""

        async def dynamic_tool_wrapper(*args, **kwargs):
            """Dynamically generated wrapper for MCP tool '{tool_name}'."""
            # added *args for unexpected types of errors i got , just a workaround for now ;)
            if args:
                logger.warning(
                    f"Tool '{tool_name}' received unexpected positional arguments: {args}. Only keyword arguments are validated."
                )

            # Validate input using the generated Pydantic model
            try:
                logger.debug(
                    f"Attempting validation for '{tool_name}' with kwargs: {kwargs}"
                )
                validated_input = InputModel(**kwargs)
                payload = validated_input.model_dump(exclude_unset=True, by_alias=True)
                logger.debug(
                    f"Validation successful for '{tool_name}'. Payload (with aliases): {payload}"
                )
            except Exception as validation_error:
                logger.error(
                    f"Input validation failed for tool '{tool_name}': {validation_error}",
                    exc_info=True,
                )
                raise ToolInputValidationError(tool_name, validation_error)

            # Get session and call the actual MCP tool
            try:
                session = self.get_session()
                logger.debug(f"Calling MCP tool '{tool_name}' with payload: {payload}")
                result = await session.call_tool(tool_name, payload)
                content = str(getattr(result, "content", ""))  # Process result
                logger.debug(f"Tool '{tool_name}' returned: {content[:100]}...")
                return content
            except (MissingConnectionException, CorruptConnectionException) as conn_err:
                logger.error(
                    f"Connection error during tool '{tool_name}' execution: {conn_err}"
                )
                raise  # Re-raise connection errors
            except Exception as call_error:
                logger.error(
                    f"Error calling tool '{tool_name}': {call_error}", exc_info=True
                )
                raise ToolExecutionError(tool_name, call_error)

        # Set function metadata
        dynamic_tool_wrapper.__name__ = sanitize_name(tool_name)
        dynamic_tool_wrapper.__doc__ = f"Dynamically generated wrapper for MCP tool: {tool_name}. Schema defined by {InputModel.__name__}."
        return dynamic_tool_wrapper

    async def create_mcp_tool(
        self,
        tool_name: str,
        tool_desc: str | None,
        input_schema: dict,
        reader_v3: SchemaReaderV3,  # V3 because i upgraded Schema reader 3 times ughh (1:too basic, 2:powerful but might induce vulnerabilities, 3:It speaks for itself ;) )
        all_definitions: Dict[str, Any],
    ) -> Callable:
        """Generates a decorated, AI-callable function for a specific MCP tool.

        Args:
            tool_name: The official tool name.
            tool_desc: The tool description.
            input_schema: The tool's input JSON schema.
            reader_v3: The SchemaReaderV3 instance (shared for caching).
            all_definitions: Global schema definitions for $ref resolution.

        Returns:
            The decorated callable async function.

        Raises:
            SchemaError: If schema processing fails.
            ToolCreationError: If wrapper/decorator creation fails.
        """
        logger.debug(f"Starting creation process for tool: '{tool_name}'")
        tool_desc = tool_desc or f"MCP tool: {tool_name} (Description unavailable)"

        # Process Schema -> Pydantic Model Object
        try:
            InputModel, _ = reader_v3.process_tool_schema(
                tool_name=tool_name,
                input_schema=input_schema,
                definitions=all_definitions,
            )
            logger.debug(
                f"Schema processed for '{tool_name}'. Generated model: {InputModel.__name__}"
            )
        except SchemaError:
            logger.error(
                f"Schema processing error for tool '{tool_name}'", exc_info=True
            )
            raise
        except Exception as e:
            logger.error(
                f"Unexpected error during schema processing for tool '{tool_name}': {e}",
                exc_info=True,
            )
            raise ToolCreationError(
                tool_name, f"Schema processing failed unexpectedly: {e}"
            ) from e

        # Create Wrapper Function Object
        try:
            wrapper_method = self._create_tool_wrapper_func(
                tool_name=tool_name, InputModel=InputModel
            )
            logger.debug(f"Wrapper function created for '{tool_name}'.")
        except Exception as e:
            logger.error(
                f"Failed to create wrapper function for tool '{tool_name}': {e}",
                exc_info=True,
            )
            raise ToolCreationError(
                tool_name, f"Wrapper function creation failed: {e}"
            ) from e

        # Apply AI Callable Decorator
        try:
            decorated_method = ai_callable(
                name=tool_name, description=tool_desc, input_model=InputModel
            )(wrapper_method)
            logger.debug(f"Applied ai_callable decorator to '{tool_name}'.")
        except Exception as e:
            logger.error(
                f"Failed to apply ai_callable decorator for tool '{tool_name}': {e}",
                exc_info=True,
            )
            raise ToolCreationError(
                tool_name, f"Decorator application failed: {e}"
            ) from e

        logger.info(
            f"Successfully created tool object for: '{tool_name}' using model {InputModel.__name__}"
        )
        return decorated_method

    async def create_function_context(self) -> FunctionContext:
        """Builds a FunctionContext containing methods for all MCP server tools.

        Fetches tools, processes schemas, creates wrappers, resolves references,
        and compiles the FunctionContext class. Logs warnings for tools that fail
        creation but does not raise an error for individual failures.

        Returns:
            A FunctionContext instance populated with available tool methods.

        Raises:
            MCPConnectionError: If listing tools from the server fails.
        """
        logger.info(
            "Fetching tools from MCP Server to build FunctionContext (SchemaReaderV3)..."
        )
        # Get tool definitions from server
        try:
            session = self.get_session()
            response = await session.list_tools()
        except Exception as e:
            logger.error(f"Failed to list tools from MCP server: {e}", exc_info=True)
            raise MCPConnectionError(f"Failed to list tools from server: {e}") from e

        methods: Dict[str, Callable] = {}
        failed_tools: List[str] = []
        successful_tools: int = 0
        reader_v3 = SchemaReaderV3()
        all_definitions = getattr(response, "definitions", {})
        tools_list = getattr(response, "tools", [])

        if not isinstance(tools_list, list):
            logger.warning(
                "Server list_tools response did not contain a valid 'tools' list."
            )
            tools_list = []

        # Process each tool definition
        for tool in tools_list:
            if not hasattr(tool, "name") or not hasattr(tool, "inputSchema"):
                logger.warning(
                    f"Skipping invalid tool entry received from server: {getattr(tool, 'name', tool)}"
                )
                continue
            tool_name = tool.name
            logger.debug(f"Processing tool definition for: '{tool_name}'")
            try:
                # Create the callable method for this tool
                method = await self.create_mcp_tool(
                    tool_name=tool_name,
                    tool_desc=getattr(tool, "description", None),
                    input_schema=tool.inputSchema,
                    reader_v3=reader_v3,
                    all_definitions=all_definitions,
                )
                # Store method using sanitized name
                methods[sanitize_name(tool_name)] = method
                successful_tools += 1
            except (SchemaError, ToolCreationError) as e:
                failed_tools.append(tool_name)
                logger.warning(
                    f"Skipping tool '{tool_name}' due to error during V3 creation: {e}",
                    exc_info=True,
                )
            except Exception as e:
                failed_tools.append(tool_name)
                logger.error(
                    f"Unexpected error skipping tool '{tool_name}': {e}", exc_info=True
                )

        # Post-Processing: Resolve Forward References
        all_created_models = reader_v3.get_created_models()
        logger.debug(
            f"Attempting to resolve forward references for {len(all_created_models)} models..."
        )
        for model_name, model in all_created_models.items():
            try:
                if isinstance(model, type) and issubclass(model, PydanticBaseModel):
                    if hasattr(model, "model_rebuild"):
                        model.model_rebuild(force=True)
            except Exception as e:
                logger.warning(
                    f"Failed to resolve forward refs for model {model_name}: {type(e).__name__} - {e}",
                    exc_info=False,
                )

        if failed_tools:
            sorted_failed_tools = sorted(failed_tools)
            logger.warning(
                f"Skipped {len(sorted_failed_tools)} tools due to errors: {', '.join(sorted_failed_tools)}"
            )

        # Create and return the FunctionContext class instance
        MCPToolClass = type("MCPFunctionContext", (FunctionContext,), methods)
        instance = MCPToolClass()
        total_tools = len(tools_list)
        logger.info(
            f"FunctionContext built with {successful_tools}/{total_tools} tools."
            f"{f' Skipped {len(failed_tools)}.' if failed_tools else ''}"
        )
        return instance
