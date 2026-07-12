import pandas as pd
import matplotlib.pyplot as plt

import pandas as pd
import matplotlib.pyplot as plt

# ===================== 全局字体配置（核心：解决中文方框问题）=====================
plt.rcParams['font.sans-serif'] = ['SimSun', 'SimHei', 'Microsoft YaHei']  # 优先加载宋体/黑体/微软雅黑
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示为方框的问题
plt.rcParams['font.size'] = 12              # 五号字对应10.5pt，这里用12pt更清晰

# ===================== 读取数据 =====================
data = pd.read_excel(r"C:\Users\GY\Desktop\xlw\pic\var.xlsx", index_col=0)

# 定义模型颜色（和你原图保持一致）
colors = {
    "TimeFusionNet": "red",
    "CNN-LSTM": "green",
    "GRU": "blue",
    "LSTM": "cyan",
    "RNN": "purple"
}

# 获取x轴标签（步骤名称）
steps = data.columns.tolist()

# ===================== 绘图 =====================
plt.figure(figsize=(10, 8))  # 稍大画布，更清晰

for model in data.index:
    values = data.loc[model].tolist()
    plt.plot(
        steps, values,
        color=colors[model],
        marker='o',       # 圆点标记
        markersize=8,     # 标记大小
        linewidth=1.5,    # 线条粗细
        label=model
    )

# ===================== 标签与图例 =====================
plt.xlabel('变量', fontsize=16)    # 替换为你需要的中文x轴名
plt.ylabel('预测误差的方差值', fontsize=16)
plt.legend(loc='upper left', fontsize=16)  # 图例位置

# 可选：添加网格（和你原图风格一致）
plt.grid(True, linestyle='--', alpha=0.3, color='gray')

# 显示图表
plt.show()


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
#
# # ================= 全局设置（关键：放大字体+优化布局）=================
# plt.rcParams['axes.unicode_minus'] = False  # 使用ASCII减号
# plt.rcParams['font.family'] = 'SimHei'      # 中文字体
# # 全局字体放大（按需调整数值）
# plt.rcParams['font.size'] = 12              # 默认字体大小（原默认10）
# plt.rcParams['axes.titlesize'] = 14         # 轴标题字体大小
# plt.rcParams['axes.labelsize'] = 12         # 轴标签字体大小
# plt.rcParams['xtick.labelsize'] = 11        # x轴刻度字体大小
# plt.rcParams['ytick.labelsize'] = 11        # y轴刻度字体大小
# plt.rcParams['legend.fontsize'] = 11        # 图例字体大小
#
# # ================= 数据读取（保持原路径，修复第四步数据引用问题）=================
# # 第一步预测指标
# df_0 = pd.read_excel(r"C:\Users\GY\Desktop\xlw\pic\20250705_085307_最好\evaluate.xlsx")
# df1_0 = pd.read_excel(r"C:\Users\GY\Desktop\xlw\models\CNN-LSTM\RESULTS\CNNLSTM_0\evaluation_metrics.xlsx")
# df2_0 = pd.read_excel(r"C:\Users\GY\Desktop\xlw\models\GRU\RESULTS\GRU_0\evaluation_metrics.xlsx")
# df3_0 = pd.read_excel(r"C:\Users\GY\Desktop\xlw\models\RNN\RESULTS\TimeFusionNetwoTA\evaluation_metrics.xlsx")
# df4_0 = pd.read_excel(r"C:\Users\GY\Desktop\xlw\models\LSTM\RESULTS\LSTM_0\evaluation_metrics.xlsx")
# TimeFusionNet_values_array_0 = df_0.loc[0:4, 'Value'].values
# CNNLSTM_values_array_0 = df1_0.loc[0:4, 'Value'].values
# GRU_values_array_0 = df2_0.loc[0:4, 'Value'].values
# RNN_values_array_0 = df3_0.loc[0:4, 'Value'].values
# LSTM_values_array_0 = df4_0.loc[0:4, 'Value'].values
#
# # 第二步预测指标
# df_1 = pd.read_excel(r"C:\Users\GY\Desktop\xlw\pic\TimeFusionNetwoSA\evaluate.xlsx")
# df1_1 = pd.read_excel(r"C:\Users\GY\Desktop\xlw\models\CNN-LSTM\RESULTS\CNNLSTM_1\evaluation_metrics.xlsx")
# df2_1 = pd.read_excel(r"C:\Users\GY\Desktop\xlw\models\GRU\RESULTS\GRU_1\evaluation_metrics.xlsx")
# df3_1 = pd.read_excel(r"C:\Users\GY\Desktop\xlw\models\RNN\RESULTS\RNN_1\evaluation_metrics.xlsx")
# df4_1 = pd.read_excel(r"C:\Users\GY\Desktop\xlw\models\LSTM\RESULTS\LSTM_1\evaluation_metrics.xlsx")
# TimeFusionNet_values_array_1 = df_1.loc[0:4, 'Value'].values
# CNNLSTM_values_array_1 = df1_1.loc[0:4, 'Value'].values
# GRU_values_array_1 = df2_1.loc[0:4, 'Value'].values
# RNN_values_array_1 = df3_1.loc[0:4, 'Value'].values
# LSTM_values_array_1 = df4_1.loc[0:4, 'Value'].values
#
# # 第三步预测指标
# df_2 = pd.read_excel(r"C:\Users\GY\Desktop\xlw\pic\20250420_145551 lag=120\工作簿2.xlsx")
# df1_2 = pd.read_excel(r"C:\Users\GY\Desktop\xlw\models\CNN-LSTM\RESULTS\CNNLSTM_2\evaluation_metrics.xlsx")
# df2_2 = pd.read_excel(r"C:\Users\GY\Desktop\xlw\models\GRU\RESULTS\GRU_2\evaluation_metrics.xlsx")
# df3_2 = pd.read_excel(r"C:\Users\GY\Desktop\xlw\models\RNN\RESULTS\RNN_2\evaluation_metrics.xlsx")
# df4_2 = pd.read_excel(r"C:\Users\GY\Desktop\xlw\models\LSTM\RESULTS\LSTM_2\evaluation_metrics.xlsx")
# TimeFusionNet_values_array_2 = df_2.loc[0:4, 'Value'].values
# CNNLSTM_values_array_2 = df1_2.loc[0:4, 'Value'].values
# GRU_values_array_2 = df2_2.loc[0:4, 'Value'].values
# RNN_values_array_2 = df3_2.loc[0:4, 'Value'].values
# LSTM_values_array_2 = df4_2.loc[0:4, 'Value'].values
#
# # 第四步预测指标（修复：原代码所有模型都用了df_3，这里按逻辑修正为对应df）
# df_3 = pd.read_excel(r"C:\Users\GY\Desktop\xlw\pic\20250420_215215 lag=180\工作簿3.xlsx")
# df1_3 = pd.read_excel(r"C:\Users\GY\Desktop\xlw\models\CNN-LSTM\RESULTS\CNNLSTM_3\evaluation_metrics.xlsx")
# df2_3 = pd.read_excel(r"C:\Users\GY\Desktop\xlw\models\GRU\RESULTS\GRU_3\evaluation_metrics.xlsx")
# df3_3 = pd.read_excel(r"C:\Users\GY\Desktop\xlw\models\RNN\RESULTS\RNN_3\evaluation_metrics.xlsx")
# df4_3 = pd.read_excel(r"C:\Users\GY\Desktop\xlw\models\LSTM\RESULTS\lstm_3\工作簿1.xlsx")
# TimeFusionNet_values_array_3 = df_3.loc[0:4, 'Value'].values
# CNNLSTM_values_array_3 = df1_3.loc[0:4, 'Value'].values  # 原df_3改为df1_3
# GRU_values_array_3 = df2_3.loc[0:4, 'Value'].values      # 原df_3改为df2_3
# RNN_values_array_3 = df3_3.loc[0:4, 'Value'].values      # 原df_3改为df3_3
# LSTM_values_array_3 = df4_3.loc[0:4, 'Value'].values      # 原df_3改为df4_3
#
# # ================= 绘图函数（修改布局+放大文字）=================
# def plot_improvement_rates(baseline_model, model2, model3, model4, model5, model_names=None, save_path=None):
#     """绘制并保存四个对比模型相对于基线模型的MAE、MSE和MAPE改进率(IR)柱状图"""
#     # 默认模型名称（修复：TimeFusionNet中间空格问题）
#     if model_names is None:
#         model_names = ["TimeFusionNet", 'CNNLSTM', "GRU", "RNN", "LSTM"]
#
#     # 模型颜色配置
#     model_colors = {
#         "CNNLSTM": "#2EF7FF",
#         "GRU": "#312FFF",
#         "RNN": "#9831FF",
#         "LSTM": "#FF8D0A"
#     }
#
#     # 数据处理
#     baseline_metrics = baseline_model
#     compare_models = [model2, model3, model4, model5]
#     compare_names = model_names[1:]
#     colors = [model_colors.get(name, "#999999") for name in compare_names]
#
#     # 计算改进率(IR)
#     def calculate_ir(baseline, compare):
#         return (abs(baseline - compare) / baseline) * 100
#
#     mae_ir = [calculate_ir(baseline_metrics[0], model[0]) for model in compare_models]
#     mse_ir = [calculate_ir(baseline_metrics[1], model[1]) for model in compare_models]
#     mape_ir = [calculate_ir(baseline_metrics[3], model[3]) for model in compare_models]
#
#     # 绘图（增大图表尺寸，原(18,6)改为(20,7)，更宽松）
#     fig, axes = plt.subplots(1, 3, figsize=(20, 7))
#     bar_width = 0.8
#     index = np.arange(len(compare_models))
#
#     # 定义柱状图数值文字大小（放大至12）
#     text_fontsize = 12
#
#     # MAE IR柱状图
#     for i in range(len(compare_models)):
#         axes[0].bar(index[i], mae_ir[i], bar_width, color=colors[i])
#     axes[0].set_title('MAE improvement rate(IR)', fontsize=20)  # 单独放大标题
#     axes[0].set_ylabel('improvement rate (%)', fontsize=14)     # 单独放大轴标签
#     axes[0].set_xticks(index)
#     axes[0].set_xticklabels(compare_names, rotation=45, ha='right', fontsize=14)
#     axes[0].axhline(y=0, color='black', linestyle='-', alpha=0.3)
#     for i, v in enumerate(mae_ir):
#         axes[0].text(i, v + 0.5 if v > 0 else v - 0.5, f'{v:.2f}%',
#                     ha='center', va='bottom' if v > 0 else 'top', fontsize=text_fontsize)
#
#     # MSE IR柱状图
#     for i in range(len(compare_models)):
#         axes[1].bar(index[i], mse_ir[i], bar_width, color=colors[i])
#     axes[1].set_title('MSE improvement rate(IR)', fontsize=20)
#     axes[1].set_ylabel('improvement rate (%)', fontsize=14)
#     axes[1].set_xticks(index)
#     axes[1].set_xticklabels(compare_names, rotation=45, ha='right', fontsize=14)
#     axes[1].axhline(y=0, color='black', linestyle='-', alpha=0.3)
#     for i, v in enumerate(mse_ir):
#         axes[1].text(i, v + 0.5 if v > 0 else v - 0.5, f'{v:.2f}%',
#                     ha='center', va='bottom' if v > 0 else 'top', fontsize=text_fontsize)
#
#     # MAPE IR柱状图
#     for i in range(len(compare_models)):
#         axes[2].bar(index[i], mape_ir[i], bar_width, color=colors[i])
#     axes[2].set_title('MAPE improvement rate(IR)', fontsize=20)
#     axes[2].set_ylabel('improvement rate (%)', fontsize=14)
#     axes[2].set_xticks(index)
#     axes[2].set_xticklabels(compare_names, rotation=45, ha='right', fontsize=14)
#     axes[2].axhline(y=0, color='black', linestyle='-', alpha=0.3)
#     for i, v in enumerate(mape_ir):
#         axes[2].text(i, v + 0.5 if v > 0 else v - 0.5, f'{v:.2f}%',
#                     ha='center', va='bottom' if v > 0 else 'top', fontsize=text_fontsize)
#
#     # 关键：优化布局（解决tight_layout警告）
#     plt.subplots_adjust(
#         bottom=0.18,  # 增大下边距（容纳x轴刻度标签）
#         top=0.88,     # 增大上边距（容纳标题）
#         left=0.05,
#         right=0.95,
#         wspace=0.3    # 增大子图之间的水平间距
#     )
#     plt.tight_layout(pad=3.0)  # 增大整体边距，彻底消除警告
#
#     # 保存图表（补充.png后缀，原路径无格式）
#     if save_path:
#         save_path = f"{save_path}.png"  # 自动添加图片后缀
#         plt.savefig(save_path, dpi=300, bbox_inches='tight')
#         print(f"图表已保存至：{save_path}")
#
#     return fig, axes
#
# # ================= 调用绘图函数（保持原逻辑）=================
# plot_improvement_rates(TimeFusionNet_values_array_0, CNNLSTM_values_array_0, GRU_values_array_0, RNN_values_array_0, LSTM_values_array_0, ["TimeFusionNet", "CNNLSTM", "GRU", "RNN", "LSTM"], "C:/Users/GY/Desktop/xlw/pic/zhuzhuangtu/0")
# plot_improvement_rates(TimeFusionNet_values_array_1, CNNLSTM_values_array_1, GRU_values_array_1, RNN_values_array_1, LSTM_values_array_1, ["TimeFusionNet", "CNNLSTM", "GRU", "RNN", "LSTM"], "C:/Users/GY/Desktop/xlw/pic/zhuzhuangtu/1")
# plot_improvement_rates(TimeFusionNet_values_array_2, CNNLSTM_values_array_2, GRU_values_array_2, RNN_values_array_2, LSTM_values_array_2, ["TimeFusionNet", "CNNLSTM", "GRU", "RNN", "LSTM"], "C:/Users/GY/Desktop/xlw/pic/zhuzhuangtu/2")
# plot_improvement_rates(TimeFusionNet_values_array_3, CNNLSTM_values_array_3, GRU_values_array_3, RNN_values_array_3, LSTM_values_array_3, ["TimeFusionNet", "CNNLSTM", "GRU", "RNN", "LSTM"], "C:/Users/GY/Desktop/xlw/pic/zhuzhuangtu/3")