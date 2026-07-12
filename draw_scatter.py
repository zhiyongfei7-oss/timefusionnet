import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import os

# ===================== 全局论文字体配置 =====================
plt.rcParams['font.sans-serif'] = ['SimSun']  # 中文：宋体
plt.rcParams['axes.unicode_minus'] = False    # 正常显示负号
plt.rcParams['font.family'] = 'sans-serif'
# 统一设置图中所有文字为 5号字（学术论文标准）
plt.rcParams['font.size'] = 10.5  # 5号字 = 10.5pt
plt.rcParams['axes.labelsize'] = 10.5
plt.rcParams['legend.fontsize'] = 10.5
plt.rcParams['xtick.labelsize'] = 10.5
plt.rcParams['ytick.labelsize'] = 10.5

def plot_scatter(y_true, y_pred, filename, time_folder):
    """
    论文规范散点图绘制函数
    :param fig_num: 图编号，如 "1-1" 表示图1-1
    :param fig_title: 图题中文（不需要加“图XX”）
    """
    plt.figure(figsize=(8, 6))

    # 确保输入是一维数组
    y_true = y_true.flatten() if len(y_true.shape) > 1 else y_true
    y_pred = y_pred.flatten() if len(y_pred.shape) > 1 else y_pred

    plt.scatter(y_true, y_pred, alpha=0.5)

    # 添加理想线
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='理想线')

    # 添加回归线
    coef = np.polyfit(y_true, y_pred, 1)
    poly1d_fn = np.poly1d(coef)
    plt.plot(y_true, poly1d_fn(y_true), 'g-', lw=2, label='回归线')

    # 添加回归方程（5号字）
    r2 = r2_score(y_true, y_pred)
    equation = f'y = {coef[0]:.4f}x + {coef[1]:.4f}\n$R^2$ = {r2:.4f}'
    plt.text(0.05, 0.95, equation, transform=plt.gca().transAxes,
             verticalalignment='top', fontsize=16,
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # 坐标轴标签（中文+5号宋体）
    plt.xlabel('真实值', fontsize=18)
    plt.ylabel('预测值', fontsize=18)
    plt.legend()

    # 保存
    scatter_path = os.path.join(time_folder, filename)
    plt.savefig(scatter_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"保存散点图到: {scatter_path}")


# ===================== 数据读取 =====================
dftrue = pd.read_excel(r"C:\Users\GY\Desktop\xlw\pic\20250705_085307_最好\predictions_20250705_085307_test.xlsx", header=1)
dfpredict_timefusionnet = pd.read_excel(r"C:\Users\GY\Desktop\xlw\pic\20250705_085307_最好\predictions_20250705_085307_test.xlsx", header=1)

test_y_original = dftrue.iloc[:, 0].values
test_pred_original_timefusionnet = dfpredict_timefusionnet.iloc[:, 1].values

# ===================== 绘图（按论文要求） =====================
plot_scatter(
    y_true=test_y_original,
    y_pred=test_pred_original_timefusionnet,
    filename="test_scatter_original_timefusionnet.png",
    time_folder='C:/Users/GY/Desktop/xlw/models'
)