from mcp.server.fastmcp import FastMCP
from utils.config import Config
from utils.task_manager import TaskManager
from utils.tasks import Task
import json

mcp = FastMCP(
    name="MCPExample",
    host=Config.Server.HOST,
    port=Config.Server.PORT,
    sse_path=Config.Server.SSE_PATH
)

manager = TaskManager()
manager.add_task("task1", is_completed=True)
manager.add_task("task2")
manager.add_task("task3")
manager.add_task("task4")

@mcp.tool()
def list_tasks(max_results: int = 10) -> list[Task]:
    tasks = manager.get_tasks()
    return tasks[:max_results]

@mcp.tool()
def add_task(name: str) -> list[Task]:
    manager.add_task(name)
    return manager.get_tasks()

@mcp.tool()
def remove_task(task_id: str) -> list[Task]:
    manager.remove_task(task_id)
    return manager.get_tasks()  
@mcp.tool()
def complete_task(task_id: str) -> list[Task]:
    manager.mark_complete(task_id)
    return manager.get_tasks()

if __name__ == "__main__":
    print(f"🚀 Starting MCP Server on {Config.Server.HOST}:{Config.Server.PORT}")
    mcp.run(transport=Config.Server.TRANSPORT)