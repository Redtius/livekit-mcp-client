import pytest
from livekit_mcp_client.connectors.base import BaseConnector

class ConcreteConnector(BaseConnector):
    async def connect(self) -> None:
        pass
    async def is_alive(self) -> bool:
        return True
    async def disconnect(self) -> None:
        pass
    def get_session(self):
        return "session_obj"

class IncompleteConnector(BaseConnector):
    async def is_alive(self) -> bool:
        return True
    async def disconnect(self) -> None:
        pass
    def get_session(self):
        return "session_obj"


def test_base_connector_cannot_instantiate():
    """Verify that the abstract BaseConnector cannot be instantiated."""
    with pytest.raises(TypeError):
        BaseConnector()

def test_concrete_connector_can_instantiate():
    """Verify that a class implementing all abstract methods can be instantiated."""
    connector = ConcreteConnector()
    assert isinstance(connector, BaseConnector)

def test_incomplete_connector_cannot_instantiate():
    """Verify that a class missing abstract methods cannot be instantiated."""
    with pytest.raises(TypeError):
        IncompleteConnector()

@pytest.mark.asyncio
async def test_concrete_connector_methods():
    """Test the methods of the concrete implementation."""
    connector = ConcreteConnector()
    await connector.connect()
    assert await connector.is_alive() is True
    await connector.disconnect()
    assert connector.get_session() == "session_obj"