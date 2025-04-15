from anyio.streams.memory import MemoryObjectReceiveStream, MemoryObjectSendStream
from typing import Tuple
from mcp.types import JSONRPCMessage

SSEClientStreams = Tuple[
    MemoryObjectReceiveStream[JSONRPCMessage | Exception],
    MemoryObjectSendStream[JSONRPCMessage],
]
