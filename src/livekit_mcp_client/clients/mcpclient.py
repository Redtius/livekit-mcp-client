import textwrap
from typing import Optional, Any, Union, Callable
from contextlib import AsyncExitStack


import logging
import asyncio
import os
import builtins

from RestrictedPython import safe_globals
from dotenv import load_dotenv
from livekit_mcp_client.clients.schema import SchemaReader
from livekit.agents.llm import (
    ai_callable,
    FunctionContext,
)
from mcp import ClientSession

from livekit_mcp_client.exceptions import (
    MCPConnectionTimeoutError,
    MCPConnectionError,
    MissingConnectionException,
    CorruptConnectionException
)
from livekit_mcp_client.connectors.base import BaseConnector
from livekit_mcp_client.utils import sanitize_name

load_dotenv()

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

server_path = os.getenv("MCP_SERVER_PATH", "mcpserver.py")


class MCPClient:
    """Client for interacting with an MCP (Modular Control Protocol) server.

    Handles connection, reconnection, and tool management with the MCP server.
    Supports both stdio and SSE protocols.
    """

    def __init__(self, connector : BaseConnector, heartbeat_interval: int = 30):
        """Initialize the MCP client.

        Args:
            protocol: The protocol to use ("stdio" or "sse")
            heartbeat_interval: Interval in seconds for heartbeat checks (default: 30)
        """
        self.exit_stack: AsyncExitStack = AsyncExitStack()

        self._connector: BaseConnector = connector
        self._is_alive: bool = False
        self._heartbeat_task: Optional[asyncio.Task] = None
        self._heartbeat_interval: int = heartbeat_interval
        self._reconnect_lock = asyncio.Lock()

    def __del__(self):
        """Destructor that warns if client wasn't properly closed."""
        if self._is_alive:
            logger.warning("MCPClient was not properly closed!")

    async def __aenter__(self):
        """Async context manager entry point.

        Returns:
            The MCPClient instance
        """
        await self.exit_stack.__aenter__()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit point.

        Handles cleanup of resources.
        """
        await self.exit_stack.__aexit__(exc_type, exc_val, exc_tb)
        
    def get_session(self)-> ClientSession:
        if self._connector is None:
            raise MissingConnectionException()
        else:
            session = self._connector.get_session()
            if session is None:
                raise CorruptConnectionException()
        return session
        
    async def _check_liveness(self) -> bool:
        """More robust check potentially including ping."""
        if not await self._connector.is_alive():
            return False
        try:
            session = self.get_session()
            await asyncio.wait_for(session.send_ping(), timeout=5.0)
            return True
        except (MissingConnectionException, CorruptConnectionException):
            logger.warning("Cannot check liveness, connection missing/corrupt.")
            return False
        except asyncio.TimeoutError:
            logger.warning("Liveness check (ping) timed out.")
            return False
        except Exception as e:
            logger.warning(f"Liveness check (ping) failed: {e}")
            return False

    async def _start_heartbeat(self) -> None:
        """Periodically check connection health."""
        while True:
            await asyncio.sleep(self._heartbeat_interval)
            try:
                if not await self._check_liveness():
                    logger.warning("Heartbeat detected dead connection, attempting reconnect...")
                    await self._reconnect()
                # else: logger.debug("Heartbeat check passed.")
            except MCPConnectionError:
                logger.error("Heartbeat failed: Reconnect failed permanently. Stopping heartbeat.")
                self._is_alive = False # Marking it as dead
                break
            except Exception as e:
                logger.error(f"Unexpected error in heartbeat loop: {e}", exc_info=True)
                logger.warning("Attempting reconnect after unexpected error...")
                try:
                    await self._reconnect()
                except MCPConnectionError:
                    logger.error("Reconnect failed after unexpected error. Stopping heartbeat.")
                    self._is_alive = False
                    break

    async def _reconnect(self, max_attempts: int = 5, base_delay: float = 1.0) -> None:
        """Reconnect with exponential backoff."""
        for attempt in range(max_attempts):
            try:
                async with self._reconnect_lock:
                    await self._connector.disconnect()
                    await self._connector.connect()
                self._is_alive = True
                logger.info(f"Reconnected after {attempt + 1} attempts")
                return
            except Exception as e:
                delay = min(base_delay * (2 ** attempt), 30)
                logger.warning(f"Reconnect attempt {attempt + 1} failed. Retrying in {delay}s...")
                await asyncio.sleep(delay)
        raise MCPConnectionError("Failed to reconnect")

    async def aclose(self) -> None:
        """Cleanup resources."""
        if self._heartbeat_task:
            self._heartbeat_task.cancel()
            try:
                await self._heartbeat_task
            except asyncio.CancelledError:
                logger.debug("Heartbeat task successfully cancelled.")
            except Exception as e:
                logger.error(f"Error awaiting cancelled heartbeat task: {e}")
            finally:
                self._heartbeat_task = None

        try:
            if self._connector:
                await self._connector.disconnect()
        except Exception as e:
            logger.error(f"Error during connector disconnect: {e}")
        finally:
            self._is_alive = False
            logger.info("MCPClient closed.")

    async def is_alive(self) -> bool:
        """Check if the connection is active."""
        return await self._connector.is_alive()
    
    async def connect(self, timeout: float = 10.0) -> None:
        """Connect with a timeout."""
        try:
            await asyncio.wait_for(self._connector.connect(), timeout)
            self._is_alive = True
            self._heartbeat_task = asyncio.create_task(self._start_heartbeat())
            logger.info("MCPClient connected successfully.")
        except asyncio.TimeoutError as e:
            self._is_alive = False
            raise MCPConnectionTimeoutError(timeout) from e
        except Exception as e:
            self._is_alive = False
            logger.error(f"MCPClient connection failed: {e}")
            raise MCPConnectionError(f"Connection failed: {e}") from e
    
    @staticmethod
    async def create_mcp_tool(tool_name: str, tool_desc: str, input_schema: dict, reader : SchemaReader):
        """Generates a dynamic AI-callable tool from JSON Schema"""
        required_params, optional_params, param_map = reader.get_parameters(input_schema)

        fn_name = sanitize_name(tool_name)
        full_params_str = ", ".join(required_params + optional_params)
        
        fn_src = textwrap.dedent(f"""
        async def {fn_name}(self, {full_params_str}):
            kwargs = {{
                {', '.join([f'"{v}": {k}' for k, v in param_map.items()])}
            }}
            kwargs = {{
                k: list(v) if isinstance(v, (tuple, set)) else v
                for k, v in kwargs.items()
                if v is not None
            }}
            result = await self.session.call_tool("{tool_name}", kwargs)
            return str(result.content)
        """)

        local_vars = {}
        try:
            exec_globals = safe_globals.copy()
            exec_globals['Any'] = Any
            exec_globals['Union'] = Union
            exec_globals['None'] = None
            exec_globals['Optional'] = Optional
            exec_globals['Callable'] = Callable
            exec_globals['list'] = builtins.list
            exec_globals['isinstance'] = builtins.isinstance
            exec_globals['tuple'] = builtins.tuple
            exec_globals['set'] = builtins.set
            exec_globals['str'] = builtins.str
            exec_globals['dict'] = builtins.dict
            code = builtins.compile(fn_src, '<inline>', 'exec')
            # code = compile_restricted(fn_src, '<inline>', 'exec')
            exec(code, exec_globals, local_vars)
            method = local_vars[fn_name]
            return ai_callable(name=tool_name, description=tool_desc)(method)
        except Exception as e:
            logger.error(f"Failed to create tool {tool_name}: {e}")
            raise ValueError(f"Tool creation failed for {tool_name}") from e

    async def create_function_context(self,reader: SchemaReader) -> FunctionContext:
        logger.info("Fetching tools from MCP Server to build FunctionContext...")
        session : ClientSession = self.get_session()
        response = await session.list_tools()
        methods = {}
        failed_tools = []
        successful_tools = 0

        for tool in response.tools:
            tool_name = tool.name
            try:
                method = await self.create_mcp_tool(
                    tool_name,
                    tool.description,
                    tool.inputSchema,
                    reader= reader
                )
                sanitized_method_name = sanitize_name(tool_name)
                methods[sanitized_method_name] = method
                successful_tools += 1
                logger.debug(f"Successfully registered tool: {tool_name} as {sanitized_method_name}")
            except Exception as e:
                failed_tools.append(tool_name)
                logger.warning(
                    f"Skipping tool '{tool_name}' due to error: {str(e)}",
                    exc_info=True
                )

        if failed_tools:
            logger.warning(
                f"Skipped {len(failed_tools)} tools due to errors: {', '.join(failed_tools)}"
            )

        MCPToolClass = type("MCPFunctionContext", (FunctionContext,), methods)
        instance = MCPToolClass()
        instance.session = session

        logger.info(f"FunctionContext built with {successful_tools}/{len(response.tools)} tools")
        return instance

# async def loading_mcp_tools(proc: JobProcess) -> tuple[FunctionContext, asyncio.Event]:
#     """Factory function to load MCP tools into a job process.

#     Args:
#         proc: The job process to attach tools to

#     Returns:
#         tuple: (FunctionContext, Event) where the event signals when loading is complete

#     Raises:
#         Exception: If any error occurs during tool loading

#     Note:
#         The created MCP client is stored in proc.userdata["mcp_client"]
#     """
#     mcp_tools_loading_done = asyncio.Event()
#     mcp_client = None
#     try:
#         mcp_client = MCPClient()
#         await mcp_client.connect_server_with_timeout(server_path)
#         proc.userdata["mcp_client"] = mcp_client
#         fcn_ctx = await mcp_client.create_function_context()

#         mcp_tools_loading_done.set()
#         return fcn_ctx, mcp_tools_loading_done

#     except Exception as e:
#         logger.error(f"Unexpected Error: {e}")

#         if mcp_client is not None:
#             await mcp_client.aclose()

#         if not mcp_tools_loading_done.is_set():
#             mcp_tools_loading_done.set()

#         raise