from .base import BaseConnector
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from contextlib import AsyncExitStack
import asyncio
from typing import Optional


class StdioConnector(BaseConnector):
    def __init__(
        self, command: str, args: list[str], env: dict[str, str] | None = None
    ):
        self._command = command
        self._args = args
        self._env = env
        self._exit_stack = AsyncExitStack()
        self._session: Optional[ClientSession] = None

    async def connect(self) -> None:
        server_params = StdioServerParameters(
            command=self._command, args=self._args, env=self._env
        )
        stdio_transport = await self._exit_stack.enter_async_context(
            stdio_client(server_params)
        )
        stdio, write = stdio_transport
        self._session = await self._exit_stack.enter_async_context(
            ClientSession(stdio, write)
        )
        await self._session.initialize()

    async def is_alive(self) -> bool:
        if not self._session:
            return False
        try:
            await asyncio.wait_for(self._session.list_tools(), timeout=2.0)
            return True
        except (asyncio.TimeoutError, ConnectionError):
            return False

    async def disconnect(self) -> None:
        await self._exit_stack.aclose()
        self._session = None

    def get_session(self):
        return self._session
