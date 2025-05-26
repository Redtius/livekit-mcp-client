import asyncio
import sys
import os
import logging
import json
from pathlib import Path
from dotenv import load_dotenv
from typing import Optional

from livekit.agents import (
    JobContext,
    WorkerOptions,
    cli,
    JobProcess,
    AutoSubscribe,
)
from livekit.agents.llm import ChatContext, FunctionContext
from livekit.agents.pipeline import VoicePipelineAgent
from livekit.plugins import silero
from livekit_mcp_client.clients.mcpclient import MCPClient
from livekit_mcp_client.connectors.stdio import StdioConnector
from livekit_mcp_client.exceptions import MCPError
from livekit.plugins import groq

load_dotenv()

#_THIS_FILE_DIR = Path(__file__).parent.resolve()
MCP_SERVER_COMMAND = os.getenv("MCP_SERVER_COMMAND", sys.executable)
#MCP_SERVER_SCRIPT = os.getenv("MCP_SERVER_SCRIPT", str(_THIS_FILE_DIR / "mcp_server.py"))
MCP_SERVER_ARGS_JSON = os.getenv("MCP_SERVER_ARGS", "[]")
MCP_SERVER_ARGS = json.loads(MCP_SERVER_ARGS_JSON)
MCP_SERVER_ENV = None

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s:%(name)s:%(message)s')

def prewarm(proc: JobProcess):
    logger.info("[PREWARM] Prewarming VAD model...")
    try:
        proc.userdata["vad"] = silero.VAD.load()
        logger.info("[PREWARM] VAD model loaded.")
    except Exception as e:
        logger.error(f"[PREWARM] Failed to load VAD model: {e}", exc_info=True)
        raise

async def entrypoint(ctx: JobContext):
    logger.info("[ENTRYPOINT] Agent job starting...")
    agent: Optional[VoicePipelineAgent] = None

    try:
        command_to_run = MCP_SERVER_COMMAND
        args_for_command = MCP_SERVER_ARGS
        print(command_to_run)
        print(args_for_command)

        if MCP_SERVER_COMMAND != sys.executable:
             script_path_to_check = None
             args_for_command = MCP_SERVER_ARGS

        logger.info(f"Initializing StdioConnector to run: {command_to_run} {' '.join(args_for_command)}")
        connector = StdioConnector(
            command=command_to_run,
            args=args_for_command,
            env=MCP_SERVER_ENV
        )
        client = MCPClient(connector=connector, heartbeat_interval=30)

        async with client:
            logger.info("Connecting MCPClient (this will start the server process)...")
            await client.connect(timeout=15)
            logger.info("MCPClient connected.")

            if not await client.is_alive():
                raise ConnectionError("MCPClient failed to establish live connection after connect.")

            logger.info("Creating FunctionContext from MCP tools...")
            fnc_ctx = await client.create_function_context()
            logger.info("FunctionContext created successfully.")

            logger.info("Connecting to LiveKit room...")
            await ctx.connect(auto_subscribe=AutoSubscribe.AUDIO_ONLY)
            logger.info("Connected to LiveKit room.")
            await ctx.wait_for_participant()
            if not fnc_ctx:
                 raise RuntimeError("FunctionContext is unexpectedly None after MCP client setup.")
            vad_plugin = ctx.proc.userdata.get("vad")
            if not vad_plugin:
                 raise RuntimeError("VAD plugin not found in userdata after prewarming.")

            initial_ctx = ChatContext().append(
                role="system",
                text=(
                    "You are a helpful voice assistant. "
                    "Use your tools to find information or perform actions when requested."
                    "Keep responses concise. Ask for clarification if needed."
                    "Don't use complexe text or ponctuation."
                ),
            )

            logger.info("Creating agent pipeline...")
            agent = VoicePipelineAgent(
                vad=vad_plugin,
                stt=groq.STT(),
                llm=groq.LLM(model="gemma2-9b-it"),
                tts=groq.TTS(voice="Cheyenne-PlayAI"),
                chat_ctx=initial_ctx,
                max_nested_fnc_calls=5,
                fnc_ctx=fnc_ctx,
            )

            logger.info("Starting agent pipeline...")
            agent.start(ctx.room)
            #await asyncio.sleep(1)
            await agent.say("Hello! How can I help you today?", allow_interruptions=True)

            logger.info("Agent running...")
            await agent.run()

    # except asyncio.CancelledError:
    #     logger.info("[ENTRYPOINT] Job cancelled.")
    # except MCPError as e:
    #      logger.error(f"[ENTRYPOINT] MCP Client Error: {e}", exc_info=True)
    except Exception as e:
        logger.error(f"[ENTRYPOINT] An unexpected error occurred: {e}", exc_info=True)
    # finally:
    #     logger.info("[ENTRYPOINT] Cleaning up...")
    #     if agent:
    #         logger.info("Closing agent pipeline...")
    #         #await agent.close()
    #     logger.info("[ENTRYPOINT] Cleanup complete.")


if __name__ == "__main__":
    cli.run_app(
        WorkerOptions(
            entrypoint_fnc=entrypoint,
            prewarm_fnc=prewarm,
            agent_name="mcp-example-agent",
            initialize_process_timeout=40.0
        )
    )