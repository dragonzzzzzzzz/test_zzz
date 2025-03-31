import matplotlib.pyplot as plt  # 导入matplotlib库，用于绘图
import numpy as np  # 导入NumPy库，用于数值计算
from scipy.interpolate import make_interp_spline  # 从scipy导入插值函数


def visualize(result, rmse):  # 定义可视化函数，接受预测结果和RMSE作为参数
    # 测试样本的真实剩余有效寿命
    true_rul = result.iloc[:, 0:1].to_numpy()  # 获取真实剩余有效寿命并转换为NumPy数组
    # 测试样本的预测剩余有效寿命
    pred_rul = result.iloc[:, 1:].to_numpy()  # 获取预测的剩余有效寿命并转换为NumPy数组

    # 创建平滑曲线
    x = np.arange(len(true_rul))  # 横坐标：样本点的索引

    # 使用B-spline进行插值来平滑曲线
    spline_true = make_interp_spline(x, true_rul.flatten(), k=3)  # k=3为三次样条插值
    spline_pred = make_interp_spline(x, pred_rul.flatten(), k=3)  # 对预测数据进行样条插值

    # 在平滑曲线插值点之间生成新的数据点
    x_new = np.linspace(x.min(), x.max(), 500)  # 在原始数据点之间生成500个插值点
    true_rul_smooth = spline_true(x_new)  # 计算真实RUL的平滑值
    pred_rul_smooth = spline_pred(x_new)  # 计算预测RUL的平滑值

    # 绘制平滑曲线
    plt.figure(figsize=(10, 6))  # 创建一个图形，设置大小为10x6英寸
    plt.axvline(x=100, c='r', linestyle='--')  # 绘制一条垂直线，表示样本点100的位置
    plt.plot(x_new, true_rul_smooth, label='Actual Data (Smoothed)')  # 绘制平滑后的真实RUL曲线
    plt.plot(x_new, pred_rul_smooth, label='Predicted Data (Smoothed)')  # 绘制平滑后的预测RUL曲线
    plt.title('RUL Prediction on CMAPSS Data')  # 设置图形标题
    plt.legend()  # 显示图例
    plt.xlabel("Samples")  # 设置x轴标签
    plt.ylabel("Remaining Useful Life")  # 设置y轴标签
    plt.savefig('Transformer({}).png'.format(rmse))  # 保存图形为PNG文件，文件名包含RMSE值
    plt.show()  # 显示图形
