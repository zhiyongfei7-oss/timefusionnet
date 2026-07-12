# from datetime import datetime
# import torch
# from main import FullModel,  plot_and_save
# from config.params import config
# from utils.trainer import Trainer
# from data.datasets import SlidingWindowDataset
# import matplotlib.pyplot as plt
# import os
# from torch.utils.data import DataLoader
#
# data_path = "/home/gzy/xlw/triple1.xlsx"
# dataset = SlidingWindowDataset(data_path,
#                                config.window_size,
#                                config.step,
#                                config.lag)
#
# train_dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)
#
# model = FullModel()  # 替换为你的模型类
# trainer = Trainer(model, config)
# model.load_state_dict(torch.load("/home/gzy/xlw/pic/20250417_100027  完整，简化最终回归层/models/best_model.pth"))
# predictions, true_data = trainer.predict(train_dataloader)
#
#
# save_dir = r"/home/gzy/xlw/pic"
# os.makedirs(save_dir, exist_ok=True)
# # 获取当前时间
# current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
# # 创建以日期小时分钟秒命名的子文件夹
# time_folder = os.path.join(save_dir, current_time)
# os.makedirs(time_folder, exist_ok=True)
# plot_and_save(true_data, predictions, "test", f"{current_time}_test", 1000, 1200, time_folder)

import matplotlib.pyplot as plt
import numpy as np

# -------------------------- 字体、字号设置 --------------------------
plt.rcParams['font.sans-serif'] = ['SimSun']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.size'] = 10.5  # 五号字

# -------------------------- 改为正方形画布 --------------------------
plt.figure(figsize=(6, 6))

# -------------------------- 原始数据 --------------------------
x = np.array([1, 2.8, 4.2, 5.8, 7.5])
y = np.array([8.5, 7.1, 5, 3.5, 2])

# -------------------------- 画黑色帕累托虚线 --------------------------
plt.plot(x, y, marker='o', linestyle='--', linewidth=0.8, color='black',
         markerfacecolor='black', markeredgecolor='black', label='帕累托最优解')

# -------------------------- 第三个点周围 5 个均匀红点 --------------------------
center_x, center_y = 4.2, 5
radius = 0.8
n_points = 5
angles = np.linspace(0, 2 * np.pi, n_points, endpoint=False)

red_x = center_x + radius * np.cos(angles)
red_y = center_y + radius * np.sin(angles)

plt.scatter(red_x, red_y, color='red', s=25, zorder=3)


# -------------------------- 点到线段的垂直投影 --------------------------
def project_point_to_segment(px, py, x1, y1, x2, y2):
    dx = x2 - x1
    dy = y2 - y1
    if dx == 0 and dy == 0:
        return x1, y1
    t = ((px - x1) * dx + (py - y1) * dy) / (dx ** 2 + dy ** 2)
    t = max(0, min(1, t))
    x_proj = x1 + t * dx
    y_proj = y1 + t * dy
    return x_proj, y_proj


# 画每条红色最短垂线
first_line = True
for rx, ry in zip(red_x, red_y):
    min_dist = np.inf
    best_proj = (0, 0)
    for i in range(len(x) - 1):
        x1, y1 = x[i], y[i]
        x2, y2 = x[i + 1], y[i + 1]
        proj_x, proj_y = project_point_to_segment(rx, ry, x1, y1, x2, y2)
        dist = np.hypot(rx - proj_x, ry - proj_y)
        if dist < min_dist:
            min_dist = dist
            best_proj = (proj_x, proj_y)

    plt.plot([rx, best_proj[0]], [ry, best_proj[1]],
             color='red', linestyle='--', linewidth=0.8,
             label='最短距离垂线' if first_line else "")
    first_line = False

# -------------------------- 关键：让 x y 轴等比例，垂线才真垂直 --------------------------
plt.axis('equal')

# -------------------------- 网格、坐标轴、图例 --------------------------
plt.grid(True, linestyle='--', alpha=0.3, color='gray')
plt.xlabel('能耗值', fontsize=15)
plt.ylabel('游离钙值', fontsize=15)
plt.legend(edgecolor='black')

plt.show()