import inspect
import typing
from functools import cached_property
from typing import Type, Dict, Any, Callable, TYPE_CHECKING
from pydantic import BaseModel as PydanticBaseModel
from livekit_mcp_client.exceptions import (
    MissingConnectionException,
    CorruptConnectionException,
    ToolInputValidationError,
    ToolExecutionError,
)
import logging
from livekit_mcp_client.utils import sanitize_name
from livekit.agents.llm.tool_context import _FunctionToolInfo

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
if TYPE_CHECKING:
    from .mcpclient import MCPClient

class MCPToolCallable(Callable):
    """
    A callable object representing a dynamically generated MCP tool.
    It exposes a dynamic __signature__ based on its associated Pydantic model,
    allowing standard introspection tools to discover its parameters.
    """

    def __init__(self, client: 'MCPClient', tool_name: str, description: str | None, InputModel: Type[PydanticBaseModel]):
        self._client = client
        self._tool_name = tool_name
        self._description = description or f"Dynamically generated wrapper for MCP tool: {tool_name}"
        self._InputModel = InputModel
        info = _FunctionToolInfo(name=self._tool_name, description=self._description)
        self.__name__ = sanitize_name(self._tool_name)
        self.__doc__ = self._description

    @cached_property
    def __signature__(self) -> inspect.Signature:
        """Dynamically generates the function signature based on the Pydantic model."""
        parameters = []
        model_fields = getattr(self._InputModel, 'model_fields', {})
        try:
             type_hints = typing.get_type_hints(self._InputModel, globalns=globals(), localns=locals())
        except NameError as e:
             logger.warning(f"Could not fully resolve type hints for {self._InputModel.__name__} due to NameError: {e}. Using Any.")
             type_hints = {}

        for name, field_info in model_fields.items():
            annotation = type_hints.get(name, field_info.annotation)
            default = inspect.Parameter.empty
            if not field_info.is_required():
                default = field_info.get_default(call_default_factory=False)

            parameters.append(
                inspect.Parameter(
                    name=name,
                    kind=inspect.Parameter.KEYWORD_ONLY,
                    default=default,
                    annotation=annotation
                )
            )
        return inspect.Signature(parameters=parameters, return_annotation=Any)

    async def __call__(self, **kwargs: Any) -> Any:
        """Executes the MCP tool call with validation."""
        try:
            logger.debug(f"Attempting validation for '{self._tool_name}' with kwargs: {kwargs}")
            validated_input = self._InputModel(**kwargs)
            payload = validated_input.model_dump(exclude_unset=True, by_alias=True)
            logger.debug(f"Validation successful for '{self._tool_name}'. Payload (with aliases): {payload}")
        except Exception as validation_error:
            logger.error(f"Input validation failed for tool '{self._tool_name}': {validation_error}", exc_info=True)
            raise ToolInputValidationError(self._tool_name, validation_error)

        try:
            session = self._client.get_session()
            logger.debug(f"Calling MCP tool '{self._tool_name}' with payload: {payload}")
            result = await session.call_tool(self._tool_name, payload)
            content = str(getattr(result, "content", ""))
            logger.debug(f"Tool '{self._tool_name}' returned: {content[:100]}...")
            return content
        except (MissingConnectionException, CorruptConnectionException) as conn_err:
            logger.error(f"Connection error during tool '{self._tool_name}' execution: {conn_err}")
            raise
        except Exception as call_error:
            logger.error(f"Error calling tool '{self._tool_name}': {call_error}", exc_info=True)
            raise ToolExecutionError(self._tool_name, call_error)

    # to make it behave correctly when used as a method on another class instance
    def __get__(self, instance, owner=None):
         return self
