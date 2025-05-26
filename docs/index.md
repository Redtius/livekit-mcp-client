# livekit-mcp-client

[![PyPI version](https://badge.fury.io/py/livekit-mcp-client.svg)](https://badge.fury.io/py/livekit-mcp-client) <!-- Replace with your actual PyPI badge if applicable -->
[![Build Status](...]()) <!-- Add your CI/CD build status badge URL -->
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0) <!-- Adjust license badge if different -->

A Python client library for interacting with a LiveKit Modular Control Protocol (MCP) server. It allows Python applications, particularly those built with [`livekit-agents`](https://github.com/livekit/agents), to dynamically discover and securely execute tools exposed by an MCP server.

The client leverages JSON Schema definitions provided by the MCP server to generate corresponding Pydantic models and type-safe callable Python functions at runtime **without** using potentially insecure `exec()` calls.

## Features

*   **Connect to MCP Server:** Supports various communication methods via pluggable connectors (e.g., `StdioConnector`).
*   **Automatic Reconnection:** Handles temporary connection drops with exponential backoff.
*   **Connection Liveness:** Optional heartbeat mechanism to monitor connection health via MCP pings.
*   **Dynamic Tool Discovery:** Fetches available tools and their schemas from the MCP server.
*   **Secure Runtime Tool Generation:**
    *   Uses `SchemaReaderV3` to process JSON Schemas.
    *   Generates Pydantic model objects directly using `pydantic.create_model`.
    *   Creates type-safe `async` wrapper functions for calling tools.
    *   **No `exec()` required**, enhancing security.
*   **`livekit-agents` Integration:** Generates a `FunctionContext` compatible with `livekit-agents`' LLM capabilities.
*   **Custom Exceptions:** Provides a clear hierarchy for error handling.

## Requirements

*   Python [e.g., 3.10+]
*   An accessible MCP Server process compatible with the chosen connector.
*   Dependencies:
    *   `livekit-agents`
    *   `pydantic` (v2 recommended)
    *   `python-dotenv` (optional, for loading `.env` files)
    *   `mcp-library` (or the specific library providing `mcp.ClientSession`)
    *   *Add any other core dependencies*

## Installation

Install the package using pip:

```bash
pip install livekit-mcp-client
```

## Basic Usage
This example demonstrates connecting to an MCP server running as a local subprocess via standard I/O and calling a dynamically generated tool.

### Summarized Example

Here's a Step By Step Guide to use this package with you AgentVoicePipeline
```python

```

### Full Example

```python
import asyncio
import sys
import os
import logging
from pathlib import Path
from dotenv import load_dotenv

from livekit.agents import (
    JobContext,
    WorkerOptions,
    cli,
    JobProcess,
    AutoSubscribe,
)
from livekit.agents.llm import ChatContext, FunctionContext
from livekit.agents.pipeline import Pipeline
from livekit.plugins import silero
from livekit_mcp_client.clients.mcpclient import MCPClient
from livekit_mcp_client.connectors.stdio import StdioConnector
from livekit_mcp_client.exceptions import MCPError
from livekit.plugins import groq, cartesia

load_dotenv()

_THIS_FILE_DIR = Path(__file__).parent.resolve()
MCP_SERVER_COMMAND = os.getenv("MCP_SERVER_COMMAND", sys.executable)
MCP_SERVER_SCRIPT = os.getenv("MCP_SERVER_SCRIPT", str(_THIS_FILE_DIR / "mcp_server.py"))
MCP_SERVER_ARGS_STR = os.getenv("MCP_SERVER_ARGS", "")
MCP_SERVER_ARGS = MCP_SERVER_ARGS_STR.split() if MCP_SERVER_ARGS_STR else []
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
    agent: Optional[Pipeline] = None

    try:
        command_to_run = MCP_SERVER_COMMAND
        args_for_command = [MCP_SERVER_SCRIPT] + MCP_SERVER_ARGS
        script_path_to_check = MCP_SERVER_SCRIPT

        if MCP_SERVER_COMMAND != sys.executable or not MCP_SERVER_SCRIPT:
             script_path_to_check = None
             args_for_command = MCP_SERVER_ARGS

        if script_path_to_check and not Path(script_path_to_check).is_file():
             raise FileNotFoundError(f"MCP Server script not found at: {script_path_to_check}")

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
            agent = Pipeline(
                vad=vad_plugin,
                stt=groq.STT(),
                llm=groq.LLM(model="gemma2-9b-it"),
                tts=cartesia.TTS(),
                chat_ctx=initial_ctx,
                fnc_ctx=fnc_ctx,
            )

            logger.info("Starting agent pipeline...")
            agent.start(ctx.room)
            await asyncio.sleep(1)
            await agent.say("Hello! How can I help you today?", allow_interruptions=True)

            logger.info("Agent running...")
            await agent.run()

    except asyncio.CancelledError:
        logger.info("[ENTRYPOINT] Job cancelled.")
    except MCPError as e:
         logger.error(f"[ENTRYPOINT] MCP Client Error: {e}", exc_info=True)
    except Exception as e:
        logger.error(f"[ENTRYPOINT] An unexpected error occurred: {e}", exc_info=True)
    finally:
        logger.info("[ENTRYPOINT] Cleaning up...")
        if agent:
            logger.info("Closing agent pipeline...")
            await agent.close()
        logger.info("[ENTRYPOINT] Cleanup complete.")


if __name__ == "__main__":
    cli.run_app(
        WorkerOptions(
            entrypoint_fnc=entrypoint,
            prewarm_fnc=prewarm,
            agent_name="mcp-example-agent",
            initialize_process_timeout=60.0
        )
    )
```

## Contributing
Contributions are welcome! Please see CONTRIBUTING.md for guidelines. [If you don't have one yet, you can add: "Please feel free to open an issue or submit a pull request."]

## License
This project is licensed under the [Apache License 2.0] - see the LICENSE file for details.

## Support
Please open an issue on GitHub for questions, bug reports, or feature requests.