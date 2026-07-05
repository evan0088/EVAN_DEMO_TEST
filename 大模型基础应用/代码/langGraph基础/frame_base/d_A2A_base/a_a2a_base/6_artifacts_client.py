# courseware/a2a_demo/test_client.py
import asyncio
from python_a2a import A2AClient

async def main():
    # 1. 初始化两个专家 Agent 的客户端
    ticket_client = A2AClient("http://127.0.0.1:5010")

    print("[主控客户端日志] 初始化完成，准备开始任务...")
    print("-" * 50)
    #预订火车票
    ticket_query = "预订一张从北京到上海的火车票"
    print(f"[主控客户端日志]预订票务 -> '{ticket_query}'")
    ticket_result = ticket_client.ask(ticket_query)
    print(f"[主控客户端日志] 收到票务预订结果: {ticket_result}")
    print("-" * 50)

    print("[主控客户端日志] 所有任务完成！")

if __name__ == "__main__":
    print("请确保 agent_server在运行 正在运行...")
    asyncio.run(main())

