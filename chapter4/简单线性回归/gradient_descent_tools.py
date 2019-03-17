# -*- coding: utf-8 -*-

"""
    作者:     Robin
    版本:     1.0
    日期:     2018/05
    文件名:    gradient_descent_tools.py
    功能：     梯度下降算法
"""
import numpy as np

def get_gradient(theta, x, y):
    """
        根据当前的参数theta计算梯度及损失值
        参数：
            - theta:    当前使用的参数
            - x:        输入的特征
            - y:        真实标签
        返回：
            - grad:     每个参数对应的梯度（以向量的形式表示）
            - cost:     损失值
    """
    m = x.shape[0]
    y_estimate = x.dot(theta)
    error = y_estimate - y
    grad = 1.0 / m * error.dot(x)
    cost = 1.0 / (2 * m) * np.sum(error ** 2)
    return grad, cost


def gradient_descent(x, y, max_iter=1500, alpha=0.01):
    """
        梯度下降算法的实现
        参数：
            - x:        输入的特征
            - y:        真实标签
            - max_iter: 最大迭代次数，默认为1500
            - alpha:    学习率，默认为0.01
        返回：
            - theta:    学习得到的最优参数
    """
    theta = np.random.randn(2)

    # 收敛阈值
    tolerance = 1e-3

    # Perform Gradient Descent
    iterations = 1

    is_converged = False
    while not is_converged:
        grad, cost = get_gradient(theta, x, y)
        new_theta = theta - alpha * grad

        # Stopping Condition
        if np.sum(abs(new_theta - theta)) < tolerance:
            is_converged = True
            print('参数收敛')

        # Print error every 50 iterations
        if iterations % 10 == 0:
            print('第{}次迭代，损失值 {:.4f}'.format(iterations, cost))

        iterations += 1
        theta = new_theta

        if iterations > max_iter:
            is_converged = True
            print('已至最大迭代次数{}'.format(max_iter))

    return theta
