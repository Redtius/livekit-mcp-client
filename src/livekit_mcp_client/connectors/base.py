from abc import ABC, abstractmethod


class BaseConnector(ABC):
    """Interface that all protocol-specific connectors must implement."""

    @abstractmethod
    async def connect(self) -> None:
        """Establish connection to the server."""
        pass

    @abstractmethod
    async def is_alive(self) -> bool:
        """Check if the connection is active."""
        pass

    @abstractmethod
    async def disconnect(self) -> None:
        """Close the connection."""
        pass

    @abstractmethod
    def get_session(self):
        """Get the underlying session object."""
        pass
