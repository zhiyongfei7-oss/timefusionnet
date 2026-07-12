import matplotlib.pyplot as plt
import numpy as np

# -------------------------- 字体、字号设置 --------------------------
plt.rcParams['font.sans-serif'] = ['SimSun']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.size'] = 10.5  # 五号字

# -------------------------- 正方形画布 --------------------------
plt.figure(figsize=(10, 6))

# -------------------------- 原始数据（黑色点） --------------------------
x = np.array([1, 2.8, 4.2, 5.8, 7.5])
y = np.array([8.5, 7.1, 5, 3.5, 2])

# 只画黑色点，不连线
plt.plot(x, y, 'o', color='black', markerfacecolor='black', markeredgecolor='black',
         label='帕累托最优解')

# -------------------------- 第三个点周围 5 个均匀红点 --------------------------
center_x, center_y = 4.2, 5
radius = 0.8
n_points = 5
angles = np.linspace(0, 2 * np.pi, n_points, endpoint=False)

red_x = center_x + radius * np.cos(angles)
red_y = center_y + radius * np.sin(angles)

plt.scatter(red_x, red_y, color='red', s=25, zorder=3)

# -------------------------- 等比例坐标轴（保证视觉上无变形） --------------------------
plt.axis('equal')

# -------------------------- 网格、坐标轴、图例 --------------------------
plt.grid(True, linestyle='--', alpha=0.3, color='gray')
plt.xlabel('能耗值', fontsize=14)
plt.ylabel('游离钙值', fontsize=14)
plt.legend(edgecolor='black')

plt.show()