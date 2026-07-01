from langgraph.graph import START,StateGraph,END

graphBuilder = StateGraph(dict)
graphBuilder.add_edge(START, END)
graph = graphBuilder.compile()

#用下面的代码生成graph的图片
import matplotlib.pyplot as plt
import matplotlib.image as mpimg  # 导入matplotlib.image用于读取图像
try:
    # 使用 Mermaid 生成图表并保存为文件
    mermaid_code = graph.get_graph().draw_mermaid_png()
    with open("graph.jpg", "wb") as f:
        f.write(mermaid_code)

    # 使用 matplotlib 显示图像
    img = mpimg.imread("graph.jpg")
    plt.imshow(img)
    plt.axis('off')  # 关闭坐标轴
    plt.show()
except Exception as e:
    print(f"An error occurred: {e}")