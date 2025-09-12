from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_core.tools import BaseTool
from utils.config import Config
from utils.tools import call_tool
from utils.logging import log_panel

SYSTEM_PROMPT = """
You're a personal productivity assistant that can manage task lists.
Your purpose is to help the user complete their productivity tasks - add task, show tasks, mark tasks as complete and delete

<instructions>
    <instruction>Always use the available tools to manage the tasks (they are stored in a database) </instruction> 
    <instruction>When displaying tasks, show the id, name and the completion status</instruction>>
    <instruction>Use the 'list_tasks' tool to get a list of tasks</instruction>
    <instruction>Use the 'add_task tool to add a new task</instruction>
    <instruction>Use the 'complete_task tool to mark a task as complete</instruction>
    <instruction>Use the 'remove_task tool to delete a task</instruction>
    <instruction>Use ✅ emoji to show completion status</instruction>
    <instruction>Use emoji ⬜ to show tasks that are not complete</instruction>
    <instruction>Use at most one tool per user query</instruction> <instruction>Never duplicate tool calls</instruction>
</instructions>
Your responses should be formatted as Markdown. Prefer using tables or lists for displaying data where appropriate.
""".strip()

def create_history() -> list[BaseChatModel]:
    return [SystemMessage(content=SYSTEM_PROMPT)]


async def ask( query: str, 
              history: list[BaseMessage], 
              llm: BaseChatModel,
              available_tools: list[BaseTool],
              max_iterations: int = Config.Agent.MAX_ITERATIONS
              ) -> str:
    log_panel(title="User Request", content=query)
    
    n_iterations = 0
    messages = history.copy()
    messages.append(HumanMessage(content=query))
    
    while n_iterations < max_iterations:
        response = await llm.ainvoke(messages)
        messages.append(response)
        
        if not response.tool_calls:
            return response.content
        
        for tool_call in response.tool_calls:
            log_panel(
                title="Tool Call",
                content=tool_call
            )
            
            response = await call_tool(tool_call, available_tools)
            messages.append(response)
            
        n_iterations += 1
        
    raise RuntimeError(
        "maximum number of iterations reached, try again with different querry"
    )
    
    
    
    
    