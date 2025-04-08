import asyncio
import inspect
import types
from typing import Optional
from contextlib import AsyncExitStack
import textwrap
import keyword
import re
from dotenv import load_dotenv
from livekit.agents.llm import (
    ChatMessage,
    ai_callable,
    LLM,
    FunctionContext
)
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

# Load environment variables
load_dotenv()

# JSON type mapper
def map_json_type_to_py(json_type: str):
    return {
        "string": str,
        "number": float,
        "integer": int,
        "boolean": bool,
        "object": dict,
        "array": list
    }.get(json_type, str)

def sanitize_name(name):
    # replace hyphens with underscores and prepend underscore if needed
    safe = re.sub(r'\W|^(?=\d)', '_', name)
    if keyword.iskeyword(safe):
        safe += '_'
    return safe


class MCPClient:
    def __init__(self):
        self.session: Optional[ClientSession] = None
        self.exit_stack = AsyncExitStack()
        
    async def __aenter__(self):
        await self.exit_stack.__aenter__()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.exit_stack.__aexit__(exc_type, exc_val, exc_tb)
        
    async def aclose(self):
        """Proper cleanup method"""
        if hasattr(self, 'exit_stack'):
            await self.exit_stack.aclose()

    async def connect_to_server(self, server_script_path: str):
        """Connect to an MCP server"""
        print(f"[INFO] Connecting to MCP server: {server_script_path}")
        
        is_python = server_script_path.endswith('.py')
        is_js = server_script_path.endswith('.js')
        if not (is_python or is_js):
            raise ValueError("Server script must be a .py or .js file")

        command = "python" if is_python else "node"
        server_params = StdioServerParameters(
            command=command,
            args=[server_script_path],
            env=None
        )

        print(f"[DEBUG] Using command: {command} {server_script_path}")
        stdio_transport = await self.exit_stack.enter_async_context(stdio_client(server_params))
        self.stdio, self.write = stdio_transport
        self.session = await self.exit_stack.enter_async_context(ClientSession(self.stdio, self.write))

        await self.session.initialize()
        print("[INFO] MCP session initialized")

        # List tools for verification
        response = await self.session.list_tools()
        tools = response.tools
        print(f"[INFO] Connected to server with tools: {[tool.name for tool in tools]}")

    async def create_mcp_tool(self, tool_name, tool_desc, input_schema):
        props = input_schema.get("properties", {})
        required = set(input_schema.get("required", []))

        # 1. Build parameter list for method definition
        param_map = {sanitize_name(k): k for k in props.keys()}

        params_str = ", ".join([
            f"{sanitized}: {map_json_type_to_py(props[orig].get('type', 'string')).__name__}"
            for sanitized, orig in param_map.items()
        ])

        fn_src = textwrap.dedent(f"""
        async def {tool_name}(self, {params_str}):
            kwargs = {{{', '.join([f'"{v}": {k}' for k, v in param_map.items()])}}}
            print("[TOOL] Calling {tool_name} with", kwargs)
            result = await self.session.call_tool("{tool_name}", kwargs)
            return str(result.content)
        """)
        
        print("[DEBUG] Generated function source:")
        print(fn_src)

        # 3. Create local namespace and exec the function
        local_vars = {}
        exec(fn_src, {}, local_vars)
        method = local_vars[tool_name]

        # 4. Decorate with ai_callable
        decorated = ai_callable(name=tool_name, description=tool_desc)(method)

        return decorated


    async def create_function_context(self) -> FunctionContext:
        print("[INFO] Fetching tools from MCP to build FunctionContext")
        response = await self.session.list_tools()
        methods = {}

        for tool in response.tools:
            print(f"[DEBUG] Registering tool: {tool.name}")
            method = await self.create_mcp_tool(tool.name, tool.description, tool.inputSchema)
            methods[tool.name] = method

        # Dynamically create a FunctionContext subclass
        MCPToolClass = type("MCPFunctionContext", (FunctionContext,), methods)

        # Bind session via instance attribute
        instance = MCPToolClass()
        instance.session = self.session  # So tool methods can use it

        print("[INFO] FunctionContext successfully built with MCP tools")
        return instance
