from typing import Any
from langchain_core.messages import ToolMessage
from langchain_core.messages.tool import ToolCall
from langchain_core.tools import (
    BaseTool, 
    StructuredTool, 
    ToolException
    )
from mcp import ClientSession
from mcp.types import (
    CallToolResult,
    TextContent
)
from mcp.types import Tool as MCPTool
import json

async def call_tool(tool_call: ToolCall, available_tools: list[BaseTool]) -> ToolMessage:
    tools_dict = {
        tool.name: tool
        for tool in available_tools
    }
    
    tool = tools_dict[tool_call["name"]]
    response = await tool.ainvoke(tool_call["args"])
    
    return ToolMessage(content=str(response), tool_call_id=tool_call["id"])

async def load_tools(session: ClientSession) -> list[BaseTool]:
    tools = await session.list_tools()
    return [_convert_mcp_to_langchain(session, tool) for tool in tools.tools]

def _convert_mcp_to_langchain(session: ClientSession, tool: MCPTool) -> BaseTool:
    async def call_tool(**args: dict[str, Any]) -> str | list[str]:
        args_json = json.dumps(args)
        call_tool_res = await session.call_tool(tool.name, args)
        return _convert_call_tool_res(call_tool_res)
    
    return StructuredTool(
        name=tool.name,
        description=tool.description or "",
        args_schema=tool.inputSchema,
        coroutine=call_tool,
    )
    
def _convert_call_tool_res(call_tool_res: CallToolResult) -> str | list[str]:
    if call_tool_res.isError:
        raise ToolException(str(call_tool_res))
    
    text_contents = [
        content for content in call_tool_res.content if isinstance(content, TextContent)
    ]
    
    if len(text_contents) == 1:
        return text_contents[0].text
    
    return [content.text for content in text_contents]