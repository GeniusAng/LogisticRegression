import matplotlib.pyplot as plt


def PCA_plot(X, y):
    # 存放三类花的x，y坐标的列表
    x_1, x_2, x_3, y_1, y_2, y_3 = [], [], [], [], [], []

    # 开始绘图
    plt.figure()
    for i in range(len(X)):
        if y[i] == 'Iris-setosa':
            x_1.append(X[i][0])
            y_1.append(X[i][1])
        elif y[i] == 'Iris-versicolor':
            x_2.append(X[i][0])
            y_2.append(X[i][1])
        elif y[i] == 'Iris-virginica':
            x_3.append(X[i][0])
            y_3.append(X[i][1])

    # 绘制散点图
    plt.scatter(x_1, y_1, c='red', marker=".", label="Iris-setosa")
    plt.scatter(x_2, y_2, c='black', marker="*", label="Iris-versicolor")
    plt.scatter(x_3, y_3, c='orange', marker="v", label="Iris-virginica")

    plt.legend(loc='best')
    plt.title("PCA降维结果图")
    plt.savefig("./picture/PCA.png")
    plt.show()
