import asyncio
from langchain_mcp_adapters.tools import load_mcp_tools
from langchain_openai import ChatOpenAI
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

# Require server script path (hardcoded for this example)
server_script = "server_stdio.py"

#配置mcp服务器启动参数
server_params = StdioServerParameters(
    command=r"C:\Users\foxba\.conda\envs\smartVoyage\python" if server_script.endswith(".py") else "node",
    args=[server_script],
)

#定义个mcp客户端
mcp_client = None

#主要的异步函数run_agent
async def run():
    global mcp_client
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            mcp_client = type("MCPClientHolder", (), {"session": session})()
            tools = await load_mcp_tools(session)
            print("tools=>",tools)
            response=await  session.call_tool("get_weather", arguments={})
            print(response)

    return

 #启动运行agent
if __name__ == "__main__":
    asyncio.run(run())