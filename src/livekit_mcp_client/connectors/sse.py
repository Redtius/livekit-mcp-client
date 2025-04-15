from .base import BaseConnector
from mcp import ClientSession
from mcp.client.sse import sse_client
from contextlib import AsyncExitStack
import asyncio
from typing import Optional


class SSEConnector(BaseConnector):
    def __init__(self, url: str):
        self._url = url
        self._exit_stack = AsyncExitStack()
        self._session: Optional[ClientSession] = None

    async def connect(self) -> None:
        streams = await self._exit_stack.enter_async_context(sse_client(url=self._url))
        session_context = ClientSession(*streams)
        self._session = await session_context.__aenter__()
        await self._session.initialize()

    async def is_alive(self) -> bool:
        if not self._session:
            return False
        try:
            await asyncio.wait_for(self._session.send_ping(), timeout=2.0)
            return True
        except (asyncio.TimeoutError, ConnectionError):
            return False

    async def disconnect(self) -> None:
        await self._exit_stack.aclose()
        self._session = None

    def get_session(self):
        return self._session
