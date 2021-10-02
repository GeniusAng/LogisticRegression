import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


class LogisticRegression(object):
    def __init__(self, W_size, n_iter=30, alpha=0.01):
        """
        对数几率回归（逻辑回归）的初始化函数
        迭代次数和学习率初始化为30和0.01，实例化逻辑回归时可以传参修改
        :param W_size: 参数W的维度，这里的 W 其实是包含了 W 和 b
        :param n_iter: 迭代次数
        :param alpha: 学习率
        """
        # 迭代次数
        self.n_iter = n_iter
        # 学习率
        self.alpha = alpha
        # 系数W
        self.W = np.zeros(W_size)

    @staticmethod
    def sigmoid(z):
        """sigmoid函数"""
        return 1 / (1 + np.exp(-z))

    def y_hat_function(self, X):
        """求y_hat的函数"""
        z = np.matmul(X, self.W)
        y_hat = self.sigmoid(z)
        return y_hat

    def costFunction(self, X, y):
        """损失函数"""
        # 求出y_hat
        y_hat = self.y_hat_function(X)
        # size为样本的个数
        size = y.shape[0]
        # 将y_hat代入公式，求出损失
        cost = np.sum(-y * np.log(y_hat) - (1 - y) * np.log(1 - y_hat)) / size
        return cost

    def gradient(self, X, y):
        """计算梯度"""
        # 求出y_hat
        y_hat = self.y_hat_function(X)
        # size为样本的个数
        size = y.shape[0]
        # 求解出损失函数对W的偏导
        dW = np.matmul(X.T, (y_hat - y)) / size
        return dW

    def BGD(self, X, y):
        """
        批量梯度下降
        :param X: 样本X
        :param y: 真实值y
        :return: 预测值y_pre，损失的集合cost_data，迭代次数
        """
        # 传入的X和y其实已经是array类型
        # 因为在主函数中做数据做处理时已经将转X和y换成为了array类型
        # 这里对X和y再次转化array类型是为了再次确认并提高函数的通用性
        X = np.array(X)
        y = np.array(y)

        # 将W初始为全0的向量，并且多加了一维数据，用来表示b
        self.W = np.zeros(X.shape[1]) * self.W

        # 创建一个列表存放计算出来的cost，用于之后的画图
        cost_data = []

        # 这里的迭代次数加一是为了能打印更多的迭代信息
        # 比如：设置n_iter为20的时候是range(21)，实际迭代了0到20共二十一次
        # 此时便能打印出第20次的迭代信息
        for i in range(self.n_iter + 1):
            # 计算梯度并对W进行迭代
            dW = self.gradient(X, y)
            self.W = self.W - self.alpha * dW
            # 计算cost并添加到cost_data列表当中
            cost = self.costFunction(X, y)
            cost_data.append(cost)
            # 每10次计算一次cost并打印一次信息
            # if i % 10 == 0:
            #     print(f'第{i}次迭代：\t损失是：{cost:.5f}，\tW与b是（前四列是W,第五列是b）：{self.W}')

        # 注释了上面每十次打印一次信息，改为只在最后打印一次，这样节省空间
        print(f'第{i}次迭代：\t损失是：{cost:.5f}，\tW与b是（前四列是W,第五列是b）：{self.W}')
        # 先算出y_hat，再将y_hat进行分类，大于0.5的为1，小于0.5的为0
        y_hat = self.y_hat_function(X)
        y_pre = np.where(y_hat >= 0.5, 1, 0)
        # 返回值cost_data和n_iter是之后绘图所需的参数
        return y_pre, cost_data, self.n_iter + 1

    @staticmethod
    def probability(y_pre, y):
        """
        :param y_pre: 预测值
        :param y: 真实值
        :return: 模型预测正确率
        """
        # 传入的y_pre是列表，要转换成array
        y_pre = np.array(y_pre)
        pro = 1 - np.sum(np.abs(y_pre - y)) / y.shape[0]
        return pro * 100


def stratifiedSample(data, ratio):
    """分层抽样函数"""
    # 把数据集一分为二
    # 前一半是第一种鸢尾花，后一半是第二种鸢尾花
    data_1 = data.iloc[0:data.shape[0] // 2]
    data_2 = data.iloc[data.shape[0] // 2:data.shape[0]]

    # sample frac:按百分比随机抽样  replace:false为不放回的抽样
    train_data_1 = data_1.sample(frac=ratio, replace=False)
    train_data_2 = data_2.sample(frac=ratio, replace=False)

    # 获取抽取出的样本对应的索引值
    train_data_1_index = train_data_1.index.to_list()
    train_data_2_index = train_data_2.index.to_list()

    # 将训练样本的索引值进行拼接
    train_data_index = train_data_1_index + train_data_2_index

    # 根据索引分割出训练样本与测试样本
    train_data = data[data.index.isin(train_data_index)]
    test_data = data[~data.index.isin(train_data_index)]

    # 将样本拼接，此时样本的前半部分为训练集，后半部分即为测试集
    data = pd.concat([train_data, test_data], axis=0)
    return data


def visualization(x, y):
    """损失函数可视化函数"""
    plt.figure(figsize=(8, 4), dpi=100)
    plt.grid(True, linestyle='--', alpha=0.8)
    plt.xlabel("迭代次数", fontdict={'size': 14})
    plt.ylabel("损失", fontdict={'size': 14})
    plt.title("损失函数图")
    plt.ylim(0, 0.7)
    plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.13)
    plt.plot(x, y, linewidth=2.5, label="cost")
    plt.legend(loc=0)
    # 保存图像要在plt.show()之前，要不然保存为空白图片
    plt.savefig("./picture/cost_picture.png")
    plt.show()
