from mcp.server.fastmcp import FastMCP
from utils.config import Config

mcp = FastMCP(
    name="MCPExample",
    host=Config.Server.HOST,
    port=Config.Server.PORT,
    sse_path=Config.Server.SSE_PATH
)

@mcp.tool()
def list_tasks(max_results: int) -> list[str]:
    return [
        "task1",
        "task2",
        "task3",
        "task4"
    ][:max_results]

if __name__ == "__main__":
    mcp.run(transport=Config.Server.TRANSPORT)

