import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline

# ==================== 字体设置 ====================
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['SimSun', 'Times New Roman']
plt.rcParams['font.size'] = 10
plt.rcParams['axes.unicode_minus'] = False
# ==================================================

# 插值点（构造所需形状）
x_points = np.array([-0.5, -0.2, -0.1, 0, 0.1, 0.2, 0.5, 0.8, 1, 1.2, 1.5, 1.8, 2, 2.2, 2.5])
y_points = np.array([2.0, 1.5, 1.0, 0.0, 1.0, 1.5, 1.8, 1.05, 1.0, 1.05, 1.5, 2.01, 2.0, 2.01, 3.0])

cs = CubicSpline(x_points, y_points, bc_type='natural')

x_dense = np.linspace(-0.6, 2.6, 1000)
y_dense = cs(x_dense)

fig, ax = plt.subplots(figsize=(10, 6))

# 绘制目标函数曲线
ax.plot(x_dense, y_dense, 'b-', linewidth=2, label='目标函数 $f(x)$')

# 标记三个解点
A = (0, cs(0))
B = (1, cs(1))
C = (2, cs(2))
ax.plot(A[0], A[1], 'ro', markersize=6, label='解 A (最优但陡峭)')
ax.plot(B[0], B[1], 'go', markersize=6, label='解 B (次优且平缓)')
ax.plot(C[0], C[1], 'mo', markersize=6, label='解 C (较差且非常平缓)')

# 点标注
ax.text(A[0]-0.1, A[1]+0.3, 'A', color='red', fontsize=10, ha='center')
ax.text(B[0]+0.1, B[1]+0.1, 'B', color='green', fontsize=10, ha='center')
ax.text(C[0]+0.1, C[1]+0.1, 'C', color='purple', fontsize=10, ha='center')

# 扰动半径
delta = 0.2

# 绘制每个解的波动范围（包含上下波动）
for point, color, label in zip([A, B, C], ['r', 'g', 'm'], ['A', 'B', 'C']):
    x_perturb = np.linspace(point[0]-delta, point[0]+delta, 200)
    y_perturb = cs(x_perturb)
    y_min = np.min(y_perturb)
    y_max = np.max(y_perturb)
    # 竖直线表示波动范围
    ax.vlines(x=point[0], ymin=y_min, ymax=y_max, colors=color, linestyles='--', linewidth=2, label=f'解 {label} 扰动范围')
    # 阴影填充（半透明）
    ax.fill_betweenx([y_min, y_max], point[0]-0.02, point[0]+0.02, color=color, alpha=0.2)

# 图例
ax.legend(loc='upper left', frameon=True)

# 坐标轴标签（中文）
ax.set_xlabel('决策变量 $x$')
ax.set_ylabel('目标函数值 $f(x)$')

# 网格
ax.grid(True, linestyle=':', alpha=0.7)

# 坐标轴范围
ax.set_xlim(-0.6, 2.6)
ax.set_ylim(-0.5, 3.5)

plt.tight_layout()
plt.show()