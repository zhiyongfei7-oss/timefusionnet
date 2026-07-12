import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os

# -------------------------- 基础配置与工具函数 --------------------------
# 创建文件夹（路径建议用相对路径，避免跨设备问题，这里保留原绝对路径）
folder_path = r"C:\Users\GY\Desktop\xlw\pic\实验结果图\xiaorong"
if not os.path.exists(folder_path):
    os.makedirs(folder_path)
    print(f"已创建文件夹：{folder_path}")

# -------------------------- 数据读取与预处理 --------------------------
# 读取Excel文件（建议将文件路径整理为列表，简化代码）
file_paths = [
    r"C:\Users\GY\Desktop\xlw\pic\20250705_085307_最好\predictions_20250705_085307_test.xlsx",  # df - TimeFusionNet
    r"C:\Users\GY\Desktop\xlw\pic\TimeFusionNetwoSA\predictions_20250419_220806_test.xlsx",  #
    r"C:\Users\GY\Desktop\xlw\pic\TimeFusionNetwoTA\test_predictions_original.xlsx"  #
]

# 读取所有数据并存储
dfs = []
for path in file_paths:
    df_temp = pd.read_excel(path)
    dfs.append(df_temp)
    print(f"读取文件：{os.path.basename(path)}，数据行数：{len(df_temp)}")

# 统一数据长度：按最短的数据集截断所有数据（核心修复点）
min_length = min(len(df) for df in dfs)
print(f"所有数据统一截断为最短长度：{min_length}")

# 提取真实值和各模型预测值（确保维度一致）
true_data = dfs[0][dfs[0].columns[0]].iloc[:min_length]  # 真实值从第一个文件取
model_data = {
    "TimeFusionNet": dfs[0][dfs[0].columns[1]].iloc[:min_length],
    "TimeFusionNet-w/o-SA": dfs[1][dfs[1].columns[1]].iloc[:min_length],
    "TimeFusionNet-w/o-TA": dfs[2][dfs[2].columns[1]].iloc[:min_length]
}

# 生成统一的x轴数据（与数据长度匹配）
x = np.arange(min_length)

# -------------------------- 绘制完整图表 --------------------------
plt.figure(figsize=(12, 6))
# 绘制真实值（黑色实线，加粗）
plt.plot(x, true_data, color='black', linewidth=2, label="True Data", zorder=5)  # zorder确保真实值在最上层
# 绘制各模型预测值
colors = ['red', 'blue', 'green']
for (model_name, data), color in zip(model_data.items(), colors):
    plt.plot(x, data, color=color, label=model_name, alpha=0.8)  # alpha调整透明度避免遮挡

# 图表美化
plt.title('Model Prediction vs True Data (Full Range)', fontsize=14, fontweight='bold')
plt.xlabel('Time Step', fontsize=18)
plt.ylabel('Value', fontsize=18)
plt.legend(loc='best', fontsize=10)
plt.grid(True, alpha=0.3)
plt.tight_layout()  # 自动调整布局，避免标签被截断
# 保存图片（使用更高分辨率）
plt.savefig(os.path.join(folder_path, "完整图表.png"), dpi=300, bbox_inches='tight')
plt.close()  # 关闭图表释放内存
print("完整图表已保存")

# -------------------------- 绘制细节图（重点区间） --------------------------
# 定义需要放大的区间（建议按时间顺序排列，增强可读性）
focus_ranges = [(100, 250), (180, 190), (300, 340), (599, 610)]

for idx, (start, end) in enumerate(focus_ranges, 1):
    # 检查区间有效性（避免超出数据范围）
    if end > min_length:
        end = min_length
        print(f"区间({start}, {end})已调整为({start}, {min_length})，避免超出数据范围")
    if start >= end:
        print(f"跳过无效区间({start}, {end})：起始值大于等于结束值")
        continue

    # 创建细节图
    plt.figure(figsize=(12, 6))
    # 绘制真实值（突出显示）
    plt.plot(x[start:end], true_data[start:end], color='black', linewidth=2, label="True Data", zorder=5)
    # 绘制各模型预测值
    for (model_name, data), color in zip(model_data.items(), colors):
        plt.plot(x[start:end], data[start:end], color=color, label=model_name, alpha=0.9)

    # 细节图美化
    plt.title(f'Model Prediction vs True Data (Range: {start}-{end})', fontsize=14, fontweight='bold')
    plt.xlabel('Time Step', fontsize=18)
    plt.ylabel('Value', fontsize=18)
    plt.legend(loc='best', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.xlim(start, end)  # 固定x轴范围，聚焦目标区间
    plt.tight_layout()
    # 保存细节图
    save_name = f"细节图_{idx}_{start}_{end}.png"
    plt.savefig(os.path.join(folder_path, save_name), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"细节图{idx}（{start}-{end}）已保存")

print("所有图表绘制完成！")