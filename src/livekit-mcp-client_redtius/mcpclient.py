import textwrap
from typing import Optional,Callable
from contextlib import AsyncExitStack

import utils
import logging
import asyncio
import os

from RestrictedPython import compile_restricted,safe_globals
from dotenv import load_dotenv
from livekit.agents import JobProcess
from livekit.agents.llm import (
    ai_callable,
    FunctionContext,
)

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.client.sse import sse_client
from . import Protocol,SSEClientStreams

from exceptions import (
    MCPConnectionTimeoutError,
    MCPConnectionError,
)

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

server_path = os.getenv("MCP_SERVER_PATH", "mcpserver.py")

class MCPClient:
    def __init__(self,protocol:Protocol = "stdio",heartbeat_interval:int = 30):
        self.session: Optional[ClientSession] = None
        self.exit_stack : AsyncExitStack = AsyncExitStack()

        if protocol not in ("stdio", "sse"):
            raise ValueError("protocol must be 'stdio' or 'sse'")

        self._protocol : Protocol = protocol
        self._streams_context : Optional[SSEClientStreams] = None
        self._is_alive: bool = False
        self._heartbeat_task: Optional[asyncio.Task] = None
        self._heartbeat_interval : int = heartbeat_interval
        
    def __del__(self):
        if self._is_alive:
            logger.warning("MCPClient was not properly closed!")

    async def __aenter__(self):
        """
        Called when entering an async with block
        Typically used to acquire resources asynchronously
        """

        await self.exit_stack.__aenter__()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """
        Called when exiting an async with block
        Handles cleanup operations
        """

        await self.exit_stack.__aexit__(exc_type, exc_val, exc_tb)

    async def aclose(self):
        """
        An explicit cleanup method that can be called directly
        Useful when not using `async with`
        """
        if self._heartbeat_task:
            self._heartbeat_task.cancel()
        try:
            await self._heartbeat_task
        except asyncio.CancelledError:
            pass
        if hasattr(self, 'exit_stack'):
            await self.exit_stack.aclose()
            
        self._is_alive = False
        logger.info("Connection closed and heartbeat stopped")
        
            
    async def _start_heartbeat(self,path):
        """Start periodic heartbeat checks"""
        while True:
            try:
                if not await self._check_alive():
                    await self._reconnect(path=path)
            except Exception as e:
                logger.error(f"Heartbeat failed: {e}")
                await self._reconnect(path=path)
            
            await asyncio.sleep(self._heartbeat_interval)
    
    async def _check_alive(self) -> bool:
        """Perform actual ping check"""
        try:
            if hasattr(self.session, 'send_ping'):
                await asyncio.wait_for(self.session.send_ping(), timeout=2.0)
            else:
                if not self.session:
                    return False
                await asyncio.wait_for(self.session.list_tools(), timeout=2.0)
            
            self._is_alive = True
            return True
        except (asyncio.TimeoutError, ConnectionError) as e:
            self._is_alive = False
            logger.warning(f"Connection check failed: {type(e).__name__}")
            return False

    async def is_alive(self) -> bool:
        """Public method to check connection status"""
        if not self.session:
            return False
        return await self._check_alive()

    async def _connect_to_server(self, path: str):
        """Connect to an MCP server"""
        logger.info(f"Connecting to MCP server: {path}")
        if self._protocol == "sse":
            logger.debug(f"Connecting to: {path}...")
            self._streams_context = await self.exit_stack.enter_async_context(sse_client(url=path))
            streams = self._streams_context

            session_context = ClientSession(*streams)
            self.session = await session_context.__aenter__()

        elif self._protocol == "stdio":
            if not os.path.exists(server_path):
                logger.warning(f"Server path does not exist: {server_path}")
            
            is_python = path.endswith('.py')
            is_js = path.endswith('.js')

            if not (is_python or is_js):
                raise ValueError("Server script must be a .py or .js file")

            command = "python" if is_python else "node"

            server_params = StdioServerParameters(
                command=command,
                args=[path],
                env=None
            )

            logger.debug(f"Using command: {command} {path}...")

            stdio_transport = await self.exit_stack.enter_async_context(stdio_client(server_params))
            self.stdio, self.write = stdio_transport
            self.session = await self.exit_stack.enter_async_context(ClientSession(self.stdio, self.write))

        await self.session.initialize()
        logger.info("MCP session initialized")

    async def connect_server_with_timeout(self,path:str,timeout: float=10.0):
        try:
            await asyncio.wait_for(self._connect_to_server(path), timeout)
            self._is_alive = True
            self._heartbeat_task = asyncio.create_task(self._start_heartbeat(path))
        except asyncio.TimeoutError:
            raise MCPConnectionTimeoutError(timeout)
        
    async def _reconnect(self,path : str,max_attempts:int = 5,base_delay:float = 1.0):
        """Attempt to reconnect with exponential backoff"""
        for attempt in range(max_attempts):
            try:
                await self.aclose()
                await self._connect_to_server(path)
                self._is_alive = True
                logger.info(f"Reconnected after {attempt + 1} attempts")
                return
            except Exception as e:
                delay = min(base_delay * (2 ** attempt), 30)
                logger.warning(f"Reconnect attempt {attempt + 1} failed. Retrying in {delay}s...")
                await asyncio.sleep(delay)
        
        raise MCPConnectionError(self._protocol)

    @staticmethod
    async def _create_mcp_tool(tool_name, tool_desc, input_schema)->Callable:
        props = input_schema.get("properties", {}) if input_schema else {}
        # TODO : taking into consideration required and optional
        # required = set(input_schema.get("required", []))

        # Building parameter list for method definition
        ## Sanitization
        param_map = {utils.sanitize_name(k): k for k in props.keys()}

        params_str = ", ".join([
            f"{sanitized}: {utils.map_json_type_to_py(props[orig].get('type', 'string')).__name__}"
            for sanitized, orig in param_map.items()
        ])

        fn_name = utils.sanitize_name(tool_name)
        ## Generating function source code as string
        fn_src = textwrap.dedent(f"""
        async def {fn_name}(self, {params_str}):
            kwargs = {{{', '.join([f'"{v}": {k}' for k, v in param_map.items()])}}}
            result = await self.session.call_tool("{tool_name}", kwargs)
            return str(result.content)
        """)

        logger.debug(f"Generated function source:\n {fn_src}")
        
        # TODO: will later do it as subprocess
        local_vars = {}
        try:
            code = compile_restricted(fn_src, '<inline>', 'exec')
            exec(code, safe_globals, local_vars)
        except Exception as e:
            logger.error(f"Error compiling/executing tool {tool_name}: {e}")
            raise
        method = local_vars[fn_name]

        # Decorating with ai_callable
        decorated = ai_callable(name=tool_name, description=tool_desc)(method)

        return decorated


    async def create_function_context(self) -> FunctionContext:
        logger.info("Fetching tools from MCP Server to build FunctionContext...")
        response = await self.session.list_tools()
        methods = {}

        for tool in response.tools:
            logger.debug(f"Registering tool: {tool.name}")
            method = await self._create_mcp_tool(tool.name, tool.description, tool.inputSchema)
            methods[tool.name] = method

        # Dynamically create a FunctionContext subclass
        MCPToolClass = type("MCPFunctionContext", (FunctionContext,), methods)

        # Binding session via instance attribute
        instance = MCPToolClass()
        
        # The session is passed to be used by generated tool's functions
        instance.session = self.session

        logger.info("FunctionContext successfully built with MCP tools")
        return instance


async def loading_mcp_tools(proc: JobProcess) -> tuple[FunctionContext, asyncio.Event]:
    """Factory function that returns both the function context and loading event.

    Args:
        proc: The job process to attach MCP tools to

    Returns:
        tuple: (FunctionContext, Event) - The created context and a loading event
              that's set when loading completes

    Raises:
        Exception: Any errors during MCP tools loading
    """
    mcp_tools_loading_done = asyncio.Event()
    mcp_client = None
    try:
        mcp_client = MCPClient()
        await mcp_client.connect_server_with_timeout(server_path)
        proc.userdata["mcp_client"] = mcp_client
        fcn_ctx = await mcp_client.create_function_context()

        mcp_tools_loading_done.set()

        return fcn_ctx, mcp_tools_loading_done

    except Exception as e:
        logger.error(f"Unexpected Error: {e}")

        if mcp_client is not None:
            await mcp_client.aclose()

        # Ensuring the event is never set on failure
        if not mcp_tools_loading_done.is_set():
            mcp_tools_loading_done.set()

        raise