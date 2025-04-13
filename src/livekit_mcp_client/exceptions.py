class MCPError(Exception):
    """Base class for all MCP client exceptions."""
    def __init__(self, message: str, *args, **kwargs):
        super().__init__(message, *args, **kwargs)

# --- Connection Related Exceptions ---

class MCPConnectionError(MCPError):
    """
    Exception type raised when the connection to a given server path fails or
    a general connection issue occurs after establishment.
    """
    def __init__(self, message: str = "MCP Server connection error.", *args, **kwargs):
        super().__init__(message, *args, **kwargs)

class MCPConnectionTimeoutError(MCPConnectionError):
    """
    Exception type raised when there's a connection timeout during initial connect.
    """
    def __init__(self, timeout: float, *args, **kwargs):
        error_msg: str = f"Failed to reach the MCP Server within {timeout} seconds"
        super().__init__(error_msg, *args, **kwargs)
        self.timeout = timeout

class CorruptConnectionException(MCPConnectionError):
    """
    Exception type raised when the connection state appears invalid after establishment
    (e.g., connector exists but cannot provide a valid session).
    """
    def __init__(self, message: str = "The established connection is corrupt. Please reconnect.", *args, **kwargs):
        super().__init__(message, *args, **kwargs)

# --- State/Usage Related Exceptions ---

class MissingConnectionException(MCPError):
    """
    Exception type raised when an operation requires an established connection, but none exists.
    """
    # Pass message and any other args/kwargs to the base class
    def __init__(self, message: str = "Connection must be established before this operation. Connect to the MCP Server first.", *args, **kwargs):
        super().__init__(message, *args, **kwargs)

# --- Protocol Specific Exceptions (Example - Needs refinement) ---

class UncompatibleProtocolError(MCPError):
    """
    Exception type raised when the protocol being used is incompatible with actions
    you're trying to do.
    """
    def __init__(self, message: str = "The operation is incompatible with the current protocol.", *args, **kwargs):
        super().__init__(message, *args, **kwargs)