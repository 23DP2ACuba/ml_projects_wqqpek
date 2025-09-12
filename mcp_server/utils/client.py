import contextlib
from mcp import ClientSession
from mcp.client.sse import sse_client
from utils.config import Config
import json

def server_url():
    return f"http://{Config.Server.HOST}:{Config.Server.PORT}{Config.Server.SSE_PATH}"

@contextlib.asynccontextmanager
async def connect_to_server(url: str = server_url()):
    """
    Connect to MCP server with proper error handling and serialization.
    """
    print(f"🔌 Connecting to MCP server at {url}")
    try:
        async with sse_client(url) as (read_stream, write_stream):
            print("✅ Connected to SSE server")
            async with ClientSession(read_stream, write_stream) as session:
                await session.initialize()
                print("✅ MCP session initialized")
                yield session
    except Exception as e:
        print(f"❌ Error connecting to server: {e}")
        if hasattr(e, 'exceptions'):
            for exc in e.exceptions:
                print(f"  - Sub-exception: {exc}")
        raise