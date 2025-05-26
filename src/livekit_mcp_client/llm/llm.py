from livekit.agents import llm
from livekit.agents.llm.tool_context import get_function_info, FunctionTool, ToolContext
from livekit_mcp_client.clients.mcpclient import MCPClient
from livekit_mcp_client.clients.mcpcallable import MCPToolCallable
from pydantic import BaseModel as PydanticBaseModel
from typing import List, Dict, Any, Type,Optional
import logging
import asyncio
from livekit.agents.llm.utils import function_arguments_to_pydantic_model

logger = logging.getLogger(__name__)

class MCPLLM(llm.LLM):
    def __init__(self, base_llm: llm.LLM, mcp_client: MCPClient,tool_ctx:ToolContext):
        super().__init__()
        self._base_llm = base_llm
        self._mcp_client = mcp_client
        self._label = f"Enhanced({getattr(base_llm, 'label', 'UnknownLLM')})"
        self._tool_ctx: ToolContext = tool_ctx
        self._tool_creation_lock = asyncio.Lock()

    def _build_api_tools_with_schema(self, mcp_tools: List[FunctionTool]) -> list[dict]:
        """Helper to generate API tool definitions using Pydantic models if available."""
        llm_api_tools = []
        if not mcp_tools:
            return llm_api_tools

        for tool in mcp_tools:
            try:
                 basic_info = get_function_info(tool)
            except Exception as e:
                 logger.warning(f"{self.label}: Failed to get function info for tool {getattr(tool,'__name__','unknown')}: {e}")
                 continue

            schema_dict = {"type": "object", "properties": {}}

            if isinstance(tool, MCPToolCallable):
                 InputModel = getattr(tool, "_InputModel", None)
                 if InputModel and issubclass(InputModel, PydanticBaseModel):
                    try:
                        schema_dict = InputModel.model_json_schema()
                        logger.debug(f"{self.label}: Generated schema for tool '{basic_info.name}' from Pydantic model.")
                    except Exception as e:
                        logger.warning(f"{self.label}: Failed to generate schema from Pydantic model for '{basic_info.name}': {e}")
                 else:
                    logger.warning(f"{self.label}: MCPToolCallable '{basic_info.name}' did not have an attached Pydantic model (_InputModel). Sending empty schema.")
            else:
                 logger.warning(f"{self.label}: Tool '{basic_info.name}' is not an MCPToolCallable. Attempting standard introspection (may be inaccurate).")
                 try:
                      temp_model = function_arguments_to_pydantic_model(tool)
                      schema_dict = temp_model.model_json_schema()
                 except Exception as e:
                      logger.error(f"{self.label}: Fallback introspection failed for '{basic_info.name}': {e}. Sending empty schema.")

            api_tool_def = {
                "type": "function",
                "function": {
                    "name": basic_info.name,
                    "description": basic_info.description or "",
                    "parameters": schema_dict,
                },
            }
            llm_api_tools.append(api_tool_def)
        return llm_api_tools


    def chat(
        self,
        *,
        chat_ctx: llm.ChatContext,
        tools: list[llm.FunctionTool] | None = None,
        **kwargs: Any
    ) -> llm.LLMStream:
        """
        Intercepts the chat call, generates schemas for MCP tools, and calls
        the base LLM with the enhanced tool definitions.
        """
        logger.debug(f"{self.label}: Intercepting chat call.")

        tool_ctx = self._tool_ctx
        mcp_tools_list = list(tool_ctx.function_tools.values()) if tool_ctx else []

        api_tools = self._build_api_tools_with_schema(mcp_tools_list)
        base_llm_args = kwargs.copy()
        base_llm_args['chat_ctx'] = chat_ctx
        logger.debug(f"Here's the base llm args 1 : {base_llm_args.keys()}")

        #base_llm_args['mcptools'] = []
        logger.debug(f"Here's the base llm args 2 : {base_llm_args.keys()}")

        # if api_tools:
        #     logger.debug(f"{self.label}: Injecting {len(api_tools)} formatted tools via extra_kwargs.")
        #     extra = base_llm_args.get('extra_kwargs', {})
        #     extra['mcptools'] = api_tools
        #     base_llm_args['extra_kwargs'] = extra
        # else:
        #     logger.debug(f"{self.label}: No MCP tools found or generated.")
        #     extra = base_llm_args.get('extra_kwargs', {})
        #     #extra.pop('tools', None) # Remove if it was somehow left from previous calls
        #     if not extra:
        #         base_llm_args.pop('extra_kwargs', None)
        #     else:
        #         base_llm_args['extra_kwargs'] = extra

        logger.debug(f"{self.label}: Calling base LLM chat: {self._base_llm.label}")
        logger.debug(f"Here's the base llm args 3 : {base_llm_args.keys()}")
        try:
            return self._base_llm.chat(mcptools = api_tools,**base_llm_args)
        except Exception as e:
            logger.error(f"{self.label}: Error calling base LLM chat method: {e}", exc_info=True)
            raise

    async def aclose(self) -> None:
         logger.debug(f"{self.label}: Closing wrapper and base LLM.")
         await self._base_llm.aclose()
         await super().aclose()