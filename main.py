import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from LogisticRegression import LogisticRegression
import LogisticRegression as LR
from MyPCA import PCA_plot

if __name__ == '__main__':
    # 设置参数
    # 设置迭代次数
    n_iter = 300
    # 设置学习率
    alpha = 0.03
    # 设置训练集的占比，此处以50%当做训练集，剩余50%当测试集为例，设置为0.5
    ratio = 0.5
    # 训练集样本的个数
    num = int(ratio * 100)

    # 导入数据
    # names为列名，分别是花萼长度、花萼宽度、花瓣长度、花瓣宽度以及鸢尾花种类
    names = ["Sepal.Length", "Sepal.Width", "Petal.Length", "Petal.Width", "Species"]
    # 导入鸢尾花数据集，并设置列名
    df = pd.read_csv("./data/iris.data", names=names)

    # 选择第一类与第三类鸢尾花做分类，故剔除第二类花
    data = df.loc[(df['Species']) != 'Iris-versicolor'].reset_index(drop=True)
    # 分层抽样
    data = LR.stratifiedSample(data, ratio)
    # 系数的个数
    W_size = data.shape[1]

    # 分割数据，前四列为属性X，最后一列为标签y
    # 使用values方法可以转换为numpy二维数组
    X = data.iloc[0:data.shape[0], 0:-1].values
    y = data.iloc[0:data.shape[0], -1].values

    # 给X矩阵最后加一列1，与W相乘之后，最后一项即为b
    X = np.column_stack((X, np.ones(X.shape[0])))

    # 更换标签y，将Iris-setosa设为1，Iris-virginica设为0
    y = np.where(y == 'Iris-setosa', 1, 0)

    # 训练集
    x_train = X[0:num]
    y_train = y[0:num].reshape(num, )  # 此处y_train是二阶矩阵，所以要reshape

    # 测试集
    x_test = X[num:100]
    y_test = y[num:100].reshape(100 - num, )

    # 实例化对数几率回归（逻辑回归）
    # 传入W_size，迭代次数，学习率
    lr = LogisticRegression(W_size, n_iter, alpha)
    # 调用模型中的"批量梯度下降"算法
    # 有三个返回值：预测值y_pre，损失的集合，迭代次数
    y_train_pre, cost_data, n_iter = lr.BGD(x_train, y_train)
    print("训练集预测值：", y_train_pre)

    # 检验模型在训练集上的分类正确率
    pre_pro_train = lr.probability(y_train_pre, y_train)
    print(f'模型在训练集上预测的正确率为{pre_pro_train:.2f}%')

    # 检验模型在测试集上的分类正确率
    y_test_pre = lr.y_hat_function(x_test)
    y_test_pre = np.where(y_test_pre >= 0.5, 1, 0)
    pre_pro_test = lr.probability(y_test_pre, y_test)
    print(f'模型在测试集上预测的正确率为{pre_pro_test:.2f}%')

    # 绘制损失函数与迭代次数的变化曲线图
    x = []
    y = cost_data
    for i in range(n_iter):
        x.append(i)

    # 调用损失函数可视化函数
    LR.visualization(x, y)

    # PCA降维分析
    # 数据处理
    PCA_X = df.iloc[0:150, 0:-1].values
    PCA_y = df.iloc[0:150, -1].values

    # 降维：4->2
    transfer = PCA(n_components=2)
    PCA_X = transfer.fit_transform(PCA_X)

    #绘制PCA结果图
    PCA_plot(PCA_X, PCA_y)
