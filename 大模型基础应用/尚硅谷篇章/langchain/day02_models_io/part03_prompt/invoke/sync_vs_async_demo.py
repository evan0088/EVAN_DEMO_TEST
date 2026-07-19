"""
同步 vs 异步 对比演示

模拟两个耗时的 LLM 调用（用 asyncio.sleep 模拟网络延迟），
对比三种写法的总耗时。
"""

import time
import asyncio

# ---------- 模拟耗时任务 ----------
def sync_task(name, seconds):
    print(f"  [同步] {name} 开始，需要 {seconds} 秒")
    time.sleep(seconds)                # 模拟 I/O 耗时
    print(f"  [同步] {name} 结束")
    return f"{name} 的结果"

async def async_task(name, seconds):
    print(f"  [异步] {name} 开始，需要 {seconds} 秒")
    await asyncio.sleep(seconds)       # 模拟 I/O 耗时（不阻塞）
    print(f"  [异步] {name} 结束")
    return f"{name} 的结果"


# ---------- 方案1：同步（阻塞）----------
def demo_sync():
    print("=" * 50)
    print("方案1：同步 — 顺序执行，阻塞等待")
    print("=" * 50)
    t0 = time.perf_counter()

    r1 = sync_task("任务A", 2)
    r2 = sync_task("任务B", 3)

    print(f"\n总耗时：{time.perf_counter() - t0:.2f} 秒")
    print(f"结果：{r1}、{r2}")
    print()


# ---------- 方案2：异步但顺序 await（不会并发）----------
async def demo_async_sequential():
    print("=" * 50)
    print("方案2：异步 + 顺序 await — 写法像同步，也是挨个执行")
    print("=" * 50)
    t0 = time.perf_counter()

    r1 = await async_task("任务A", 2)
    r2 = await async_task("任务B", 3)

    print(f"\n总耗时：{time.perf_counter() - t0:.2f} 秒")
    print(f"结果：{r1}、{r2}")
    print()


# ---------- 方案3：异步 + 并发（真正的并发）----------
async def demo_async_concurrent():
    print("=" * 50)
    print("方案3：异步 + asyncio.gather — 真正的并发执行")
    print("=" * 50)
    t0 = time.perf_counter()

    # 同时创建两个任务
    task3 = await async_task("任务A", 5)

    task1 = asyncio.create_task(async_task("任务B", 2))
    task2 = asyncio.create_task(async_task("任务C", 3))


    # 一起等结果
    r1, r2 = await asyncio.gather(task1, task2)

    print(f"\n总耗时：{time.perf_counter() - t0:.2f} 秒")
    print(f"结果：{r1}、{r2}")
    print()


# ---------- 主入口 ----------
if __name__ == "__main__":
    # 先跑同步
    # demo_sync()

    # 再跑异步顺序
    # asyncio.run(demo_async_sequential())
    #
    # # 再跑异步并发
    asyncio.run(demo_async_concurrent())
    #
    # # 总结
    # print("=" * 50)
    # print("结论：")
    # print("  方案1 (同步)         : 2 + 3 = 5 秒 — 阻塞等待")
    # print("  方案2 (异步顺序await): 2 + 3 = 5 秒 — 写法异步，但还是挨个等")
    # print("  方案3 (异步并发)     : max(2, 3) ≈ 3 秒 — 两个任务同时跑")
    # print("=" * 50)
