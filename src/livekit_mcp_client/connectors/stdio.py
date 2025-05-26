# --- START OF FILE connectors/stdio.py ---

from .base import BaseConnector
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from contextlib import AsyncExitStack
import asyncio
from typing import Optional,Dict
import logging # Import logging module

# Import a potential base error if wrapping exceptions
# from ..exceptions import ConnectorConnectionError, MCPError # Example

# Get a logger specific to this module
logger = logging.getLogger(__name__)

class StdioConnector(BaseConnector):
    """
    A connector that manages and communicates with an MCP server process
    via its standard input and standard output streams.
    """
    def __init__(
        self,
        command: str,
        args: list[str],
        env: Optional[Dict[str, str]] = None, # Use Optional from typing
    ):
        """
        Initializes the StdioConnector.

        Args:
            command: The command or executable path to run the server process.
            args: A list of string arguments to pass to the command.
            env: Optional dictionary of environment variables for the subprocess.
                 If None, the current environment is inherited.
        """
        self._command = command
        self._args = args
        self._env = env
        self._exit_stack = AsyncExitStack() # Manages context of transport and session
        self._session: Optional[ClientSession] = None
        self._process = None # Store process reference if needed (e.g., for stderr reading)

        # Log initialization details at DEBUG level
        logger.debug(
            f"StdioConnector initialized: Command='{self._command}', "
            f"Args={self._args}, Env specified={'Yes' if self._env else 'No'}"
        )

    async def connect(self) -> None:
        """
        Starts the MCP server subprocess and establishes a ClientSession.

        Raises:
            ConnectorConnectionError: If any step in starting the process or session fails.
            # Or raises the specific underlying exception.
        """
        logger.info(f"Attempting to connect via stdio: command='{self._command}', args={self._args}")
        if self._session:
             logger.warning("Connect called while already connected or connecting. Disconnecting first.")
             await self.disconnect() # Ensure clean state

        try:
            # Prepare parameters for the stdio_client
            server_params = StdioServerParameters(
                command=self._command, args=self._args, env=self._env
            )
            logger.debug(f"Prepared StdioServerParameters.")

            # Start the server process and get transport (stdio streams)
            # stdio_client returns an async context manager
            logger.debug("Entering stdio_client context manager (starts process)...")
            # stdio_transport type might be Tuple[StreamReader, StreamWriter, Process] or similar
            stdio_transport = await self._exit_stack.enter_async_context(
                stdio_client(server_params)
            )
            # Extract streams and potentially the process object
            # The exact return structure depends on the mcp-library version
            # Assuming a structure like (stdio_reader, stdio_writer, process) or just (reader, writer)
            if len(stdio_transport) == 3: # Example check
                 stdio_reader, stdio_writer, self._process = stdio_transport
                 logger.debug(f"stdio_client started process with PID: {self._process.pid if self._process else 'Unknown'}")
            elif len(stdio_transport) == 2:
                 stdio_reader, stdio_writer = stdio_transport
                 self._process = None # Process handle might not be exposed
                 logger.debug("stdio_client started process (PID not directly available).")
            else:
                 raise ValueError(f"Unexpected return structure from stdio_client: {stdio_transport}")

            logger.debug("stdio_client context entered successfully.")

            # Create and initialize the MCP ClientSession using the streams
            logger.debug("Creating and entering ClientSession context manager...")
            # ClientSession is also an async context manager
            self._session = await self._exit_stack.enter_async_context(
                ClientSession(stdio_reader, stdio_writer) # Pass reader/writer streams
            )
            logger.debug("ClientSession context entered. Initializing session...")
            # Perform MCP initialization handshake
            await self._session.initialize()
            logger.info("StdioConnector connection and session initialization successful.")

        except Exception as e:
            logger.error(f"StdioConnector connection failed: {type(e).__name__} - {e}", exc_info=True)
            # Ensure cleanup happens even if connection fails mid-way
            await self._exit_stack.aclose()
            self._session = None
            self._process = None
            # Option 1: Re-raise the original exception
            raise
            # Option 2: Wrap in a custom connector error
            # raise ConnectorConnectionError(f"Failed to connect via stdio: {e}") from e

    async def is_alive(self) -> bool:
        """
        Checks if the connection is likely alive.

        Verifies session existence and attempts a lightweight MCP command (`list_tools`).

        Returns:
            True if the connection appears alive, False otherwise.
        """
        logger.debug("Checking StdioConnector liveness...")
        if not self._session:
            logger.debug("Liveness check: No active session.")
            return False
        try:
            # Use list_tools as a more comprehensive check than just ping
            logger.debug("Attempting list_tools for liveness check...")
            await asyncio.wait_for(self._session.list_tools(), timeout=2.0)
            logger.debug("Liveness check via list_tools successful.")
            return True
        except asyncio.TimeoutError:
            logger.warning("Liveness check failed: list_tools timed out.")
            return False
        # Catch connection errors specifically if mcp-library raises them
        except ConnectionError as e: # Or more specific MCP exceptions if available
             logger.warning(f"Liveness check failed: ConnectionError during list_tools: {e}")
             return False
        except Exception as e:
            # Catch other unexpected errors during the check
            logger.warning(f"Liveness check failed: Unexpected error during list_tools: {e}", exc_info=False) # Less verbose traceback
            return False

    async def disconnect(self) -> None:
        """
        Disconnects the session and terminates the managed subprocess gracefully.
        """
        logger.info("Disconnecting StdioConnector...")
        try:
            # aclose() cleans up contexts entered with enter_async_context,
            # including ClientSession and the stdio_client transport (which terminates the process).
            await self._exit_stack.aclose()
            logger.info("StdioConnector disconnected successfully (exit stack closed).")
        except Exception as e:
            logger.error(f"Error during StdioConnector disconnect: {e}", exc_info=True)
            # Optionally raise a cleanup error
            # raise ConnectorCleanupError(f"Failed during disconnect: {e}") from e
        finally:
            # Ensure session and process references are cleared
            self._session = None
            self._process = None
            logger.debug("Connector session and process reference cleared.")
            # Reset the exit stack for potential reuse (though usually connector is discarded)
            self._exit_stack = AsyncExitStack()


    def get_session(self) -> Optional[ClientSession]:
        """
        Returns the active ClientSession instance, or None if not connected.
        """
        logger.debug(f"get_session called. Returning session: {'Set' if self._session else 'None'}")
        return self._session

# --- END OF FILE connectors/stdio.py ---