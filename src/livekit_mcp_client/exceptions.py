from typing import Optional, Any

# --- Base Exceptions ---


class MCPError(Exception):
    """Base class for all MCP client exceptions."""

    def __init__(self, message: str, *args, **kwargs):
        super().__init__(message, *args, **kwargs)


class SchemaError(MCPError):
    """Base class for errors related to processing JSON schemas."""

    def __init__(self, message: str, *args, **kwargs):
        super().__init__(message, *args, **kwargs)


# --- Cleanup Exceptions ---


class MCPCleanupError(MCPError):
    """
    Exception Type Raised When The Clean up process wasn't done correctly
    """

    def __init__(self, message: str = "MCP Client wasn't closed properly"):
        super().__init__(message)


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


class MCPReconnectError(MCPConnectionError):
    """Raised specifically when the automatic reconnection logic fails after all attempts."""

    def __init__(self, message="Failed to reconnect after multiple attempts"):
        super().__init__(message)


class CorruptConnectionException(MCPConnectionError):
    """
    Exception type raised when the connection state appears invalid after establishment
    (e.g., connector exists but cannot provide a valid session).
    """

    def __init__(
        self,
        message: str = "The established connection is corrupt. Please reconnect.",
        *args,
        **kwargs,
    ):
        super().__init__(message, *args, **kwargs)


class MissingConnectionException(MCPError):
    """
    Exception type raised when an operation requires an established connection, but none exists.
    """

    def __init__(
        self,
        message: str = "Connection must be established before this operation. Connect to the MCP Server first.",
        *args,
        **kwargs,
    ):
        super().__init__(message, *args, **kwargs)


# --- Tool Related Exceptions ---


class MCPToolError(MCPError):
    """Base class for errors related to MCP tool definition or execution."""

    def __init__(self, tool_name: Optional[str] = None, message: str = ""):
        self.tool_name = tool_name
        base_msg = f"Tool '{tool_name}': {message}" if tool_name else message
        super().__init__(base_msg)


class SchemaProcessingError(MCPToolError):
    """Raised when processing a tool's JSON schema fails (e.g., invalid schema, $ref error)."""

    def __init__(self, tool_name: str, schema_error: str):
        self.schema_error = schema_error
        super().__init__(tool_name, f"Schema processing failed - {schema_error}")


class ToolCreationError(MCPToolError):
    """Raised during the creation of the tool's wrapper function or applying decorators."""

    def __init__(self, tool_name: str, detail: str):
        super().__init__(tool_name, f"Tool creation failed - {detail}")


class ToolInputValidationError(MCPToolError):
    """Raised by the tool wrapper if input arguments fail Pydantic validation."""

    def __init__(self, tool_name: str, validation_error: Exception):
        self.validation_error = validation_error
        super().__init__(tool_name, f"Input validation failed - {validation_error}")


class ToolExecutionError(MCPToolError):
    """Raised by the tool wrapper if the underlying session.call_tool fails."""

    def __init__(self, tool_name: str, cause: Exception):
        super().__init__(tool_name, f"Execution failed - {cause}")
        self.__cause__ = cause


# --- Schema Related Exceptions ---


class InvalidSchemaTypeError(SchemaError):
    """Raised when the input schema is not of the expected type (e.g., not an object)."""

    def __init__(self, expected_type: str, actual_type: Any, context: str = ""):
        self.expected_type = expected_type
        self.actual_type = type(actual_type).__name__
        self.context = context
        message = (
            f"Invalid schema type: Expected '{expected_type}', got '{self.actual_type}'"
        )
        if context:
            message += f" (Context: {context})"
        super().__init__(message)


class InvalidSchemaStructureError(SchemaError):
    """Raised when a required part of the schema structure is missing or invalid (e.g., 'properties' not a dict)."""

    def __init__(
        self,
        structure_element: str,
        expected_type: str,
        actual_type: Any,
        context: str = "",
    ):
        self.structure_element = structure_element
        self.expected_type = expected_type
        self.actual_type = type(actual_type).__name__
        self.context = context
        message = f"Invalid schema structure: Element '{structure_element}' should be '{expected_type}', got '{self.actual_type}'"
        if context:
            message += f" (Context: {context})"
        super().__init__(message)


class SchemaRefResolutionError(SchemaError):
    """Raised when a JSON schema $ref pointer cannot be resolved."""

    def __init__(self, ref_path: str, context: str = ""):
        self.ref_path = ref_path
        self.context = context
        message = f"Failed to resolve schema reference: '{ref_path}'"
        if context:
            message += f" (Context: {context})"
        super().__init__(message)


class SchemaUnsupportedTypeError(SchemaError):
    """Raised when encountering a JSON schema type or format that SchemaReaderV3 doesn't handle."""

    def __init__(
        self, json_type: Any, json_format: Optional[str] = None, context: str = ""
    ):
        self.json_type = json_type
        self.json_format = json_format
        self.context = context
        message = f"Unsupported JSON schema type/format encountered: type='{json_type}', format='{json_format}'"
        if context:
            message += f" (Context: {context})"
        super().__init__(message)


class SchemaCircularReferenceError(SchemaError):
    """Raised specifically when a problematic circular reference is detected during processing."""

    def __init__(self, model_name: str, context: str = ""):
        self.model_name = model_name
        self.context = context
        message = f"Circular reference detected involving model '{model_name}'"
        if context:
            message += f" (Context: {context})"
        super().__init__(message)
