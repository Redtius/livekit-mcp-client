

class MCPConnectionTimeoutError(Exception):
  """
  Exception type raised when there's a connection timeout
  """

  def __init__(self, timeout:float):
    error_msg:str = f"Failed To Reach the MCP Server After {timeout} seconds"
    super().__init__(error_msg)
    
class MCPConnectionError(Exception):
  """
  Exception type raised when the connection to a given server path fails
  """
  def __init__(self,protocol):
    error_msg:str = f"Server Unreachable, Verify Server Path"
    super().__init__(error_msg)