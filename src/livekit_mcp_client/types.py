from anyio.streams.memory import MemoryObjectReceiveStream,MemoryObjectSendStream
from typing import Literal,Tuple
from mcp.types import JSONRPCMessage

Protocol = Literal["stdio","sse"]
SSEClientStreams = Tuple[
    MemoryObjectReceiveStream[JSONRPCMessage | Exception],
    MemoryObjectSendStream[JSONRPCMessage]
]
