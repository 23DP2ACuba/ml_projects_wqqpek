import asyncio
from dotenv import load_dotenv
from mcp import ClientSession
from mcp.client.sse import sse_client
from rich.pretty import pprint
from utils.client import server_url

load_dotenv()

async def main():
    async with sse_client(server_url()) as (read_stream, write_stream):
        async with ClientSession(read_stream, write_stream) as session:
            await session.initialize()
            res = await session.list_tools()
            
            # pprint(res.tools)
            
            res = await session.call_tool("list_tasks", arguments={"max_results": 2})
            pprint(res.content)
            
if __name__ == "__main__":
    asyncio.run(main())