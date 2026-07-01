def showGraphInCode(graph,fileName):
    import matplotlib.pyplot as plt
    import matplotlib.image as mpimg  # 导入matplotlib.image用于读取图像
    try:
        # 使用 Mermaid 生成图表并保存为文件
        mermaid_code = graph.get_graph().draw_mermaid_png()
        with open(fileName, "wb") as f:
            f.write(mermaid_code)

        # 使用 matplotlib 显示图像
        img = mpimg.imread(fileName)
        plt.imshow(img)
        plt.axis('off')  # 关闭坐标轴
        plt.show()
    except Exception as e:
        print(f"An error occurred: {e}")

def showGraphInCode2(graph):
    import matplotlib.pyplot as plt
    import matplotlib.image as mpimg  # 导入matplotlib.image用于读取图像
    try:
        # 使用 Mermaid 生成图表并保存为文件
        print("before mermid")
        mermaid_code = graph.get_graph().draw_png()
        print("after mermid")
        with open("graph.jpg", "wb") as f:
            f.write(mermaid_code)

        # 使用 matplotlib 显示图像
        img = mpimg.imread("graph.jpg")
        plt.imshow(img)
        plt.axis('off')  # 关闭坐标轴
        plt.show()
    except Exception as e:
        print(f"An error occurred: {e}")

"""def showGraphInJupyter(graph):
    from IPython.display import Image, display
    try:
        display(Image(graph.get_graph().draw_mermaid_png()))
    except Exception:
        # This requires some extra dependencies and is optional
        print("error")
        pass"""