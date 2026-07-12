import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os

# -------------------------- 全局设置（论文图表规范） --------------------------
# 设置中文字体为黑体（SimHei）作为备选，英文字体为Times New Roman
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Times New Roman']  # 添加通用黑体
plt.rcParams['font.family'] = 'sans-serif'  # 使用sans-serif族（中文字体多属此类）
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
plt.rcParams['mathtext.fontset'] = 'cm'  # 数学公式字体

# 图表样式参数（统一为10pt）
plt.rcParams['axes.linewidth'] = 1.0
plt.rcParams['xtick.major.width'] = 1.0
plt.rcParams['ytick.major.width'] = 1.0
plt.rcParams['lines.linewidth'] = 1.5
plt.rcParams['legend.frameon'] = True
plt.rcParams['legend.framealpha'] = 1.0
plt.rcParams['legend.fontsize'] = 10      # 图例10pt
plt.rcParams['axes.labelsize'] = 10       # 坐标轴标签10pt
plt.rcParams['xtick.labelsize'] = 10      # x轴刻度10pt
plt.rcParams['ytick.labelsize'] = 10      # y轴刻度10pt


# -------------------------- 数据读取与预处理 --------------------------
folder_path = r"C:\Users\GY\Desktop\xlw\pic\实验结果图\xiaorong"
if not os.path.exists(folder_path):
    os.makedirs(folder_path)

try:
    df = pd.read_excel(r"C:\Users\GY\Desktop\xlw\pic\20250705_085307_最好\predictions_20250705_085307_test.xlsx")
    df1 = pd.read_excel(r"C:\Users\GY\Desktop\xlw\pic\TimeFusionNet_NoLag\predictions_20260317_153618_test.xlsx")
    #df2 = pd.read_excel(r"C:\Users\GY\Desktop\xlw\models\RNN\RESULTS\20250717_100240\test_predictions_original.xlsx")
except FileNotFoundError as e:
    print(f"文件读取失败：{e}")
    exit(1)

True_data = df.columns[0] if len(df.columns) > 0 else "True Value"
TimeFusionNet = df.columns[1] if len(df.columns) > 1 else "TimeFusionNet"
CNNLSTM = df1.columns[1] if len(df1.columns) > 1 else "CNN-LSTM"
#GRU = df2.columns[1] if len(df2.columns) > 1 else "GRU"

x = np.arange(len(df))
y1 = df[True_data].values
y2 = df[TimeFusionNet].values
y3 = df1[CNNLSTM].values if len(df1) >= len(df) else df1[CNNLSTM].values[:len(df)]
#y4 = df2[GRU].values if len(df2) >= len(df) else df2[GRU].values[:len(df)]

min_len = min(len(x), len(y1), len(y2), len(y3))
x = x[:min_len]
y1 = y1[:min_len]
y2 = y2[:min_len]
y3 = y3[:min_len]
#y4 = y4[:min_len]

# -------------------------- 绘制完整图表（无图题，中文坐标轴） --------------------------
fig, ax = plt.subplots(figsize=(12, 6), dpi=300)

ax.plot(x, y1, color='black', linestyle='-', label='True Value', zorder=5)
ax.plot(x, y2, color='#d62728', linestyle='-', label='TimeFusionNet', zorder=4)
ax.plot(x, y3, color='#1f77b4', linestyle='--', label='CNN-LSTM', zorder=3)
#ax.plot(x, y4, color='#2ca02c', linestyle='-.', label='GRU', zorder=2)

# 设置中文坐标轴标签（字号已全局设为10）
ax.set_xlabel('Time Step')
ax.set_ylabel('Value')
# 已删除 ax.set_title()

# 图例保持英文，位置右上角（可根据需要调整）
ax.legend(loc='upper right', bbox_to_anchor=(1, 1), ncol=2, frameon=True)

ax.grid(axis='y', linestyle=':', alpha=0.7)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()

fig.savefig(os.path.join(folder_path, "完整图表.png"), dpi=300, bbox_inches='tight')
fig.savefig(os.path.join(folder_path, "完整图表.eps"), format='eps', bbox_inches='tight')
plt.close(fig)

# -------------------------- 绘制细节图（同样无图题，中文坐标轴） --------------------------
ranges = [(100, 250), (300, 340), (180, 190), (499, 510)]

for i, (start, end) in enumerate(ranges):
    start = max(0, start)
    end = min(len(x), end)
    if start >= end:
        print(f"区间({start}-{end})无效，跳过")
        continue

    fig, ax = plt.subplots(figsize=(8, 4), dpi=300)
    ax.plot(x[start:end], y1[start:end], color='black', linestyle='-', label='True Value', zorder=5)
    ax.plot(x[start:end], y2[start:end], color='#d62728', linestyle='-', label='TimeFusionNet', zorder=4)
    ax.plot(x[start:end], y3[start:end], color='#1f77b4', linestyle='--', label="TimeFusionNet_NoLag", zorder=3)
    #ax.plot(x[start:end], y4[start:end], color='#2ca02c', linestyle='-.', label="TimeFusionNet-w/o-TA", zorder=2)

    ax.set_xlabel('时间', fontsize=12)
    ax.set_ylabel('游离钙值', fontsize=12)
    # 无标题

    # 细节图图例使用最佳位置（可自行调整）
    ax.legend(loc='best', frameon=True)

    ax.grid(axis='y', linestyle=':', alpha=0.7)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()

    fig.savefig(os.path.join(folder_path, f"细节图_{start}_{end}.png"), dpi=300, bbox_inches='tight')
    fig.savefig(os.path.join(folder_path, f"细节图_{start}_{end}.eps"), format='eps', bbox_inches='tight')
    plt.close(fig)

print("图表绘制完成，已保存至：", folder_path)