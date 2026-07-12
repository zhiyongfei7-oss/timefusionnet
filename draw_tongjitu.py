import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
# 设置全局参数（在导入pyplot后立即设置）
plt.rcParams['axes.unicode_minus'] = False  # 使用ASCII减号代替Unicode减号
plt.rcParams['font.family'] = 'SimHei'      # 保持中文字体

# 读取Excel文件
#第一步预测指标
df_0 = pd.read_excel(r"C:\Users\GY\Desktop\xlw\pic\20250705_085307_最好\evaluate.xlsx")
df1_0 = pd.read_excel(r"C:\Users\GY\Desktop\xlw\models\CNN-LSTM\RESULTS\CNNLSTM_0\evaluation_metrics.xlsx")
df2_0 = pd.read_excel(r"C:\Users\GY\Desktop\xlw\models\GRU\RESULTS\GRU_0\evaluation_metrics.xlsx")
df3_0 = pd.read_excel(r"C:\Users\GY\Desktop\xlw\models\RNN\RESULTS\RNN_0\evaluation_metrics.xlsx")
df4_0 = pd.read_excel(r"C:\Users\GY\Desktop\xlw\models\LSTM\RESULTS\LSTM_0\evaluation_metrics.xlsx")
TimeFusionNet_values_array_0 = df_0.loc[0:4, 'Value'].values
CNNLSTM_values_array_0 = df1_0.loc[0:4, 'Value'].values
GRU_values_array_0 = df2_0.loc[0:4, 'Value'].values
RNN_values_array_0 = df3_0.loc[0:4, 'Value'].values
LSTM_values_array_0 = df4_0.loc[0:4, 'Value'].values


#第二步预测指标
df_1 = pd.read_excel(r"C:\Users\GY\Desktop\xlw\pic\20250419_220806 lag=60\evaluate.xlsx")
df1_1 = pd.read_excel(r"C:\Users\GY\Desktop\xlw\models\CNN-LSTM\RESULTS\CNNLSTM_1\evaluation_metrics.xlsx")
df2_1 = pd.read_excel(r"C:\Users\GY\Desktop\xlw\models\GRU\RESULTS\GRU_1\evaluation_metrics.xlsx")
df3_1 = pd.read_excel(r"C:\Users\GY\Desktop\xlw\models\RNN\RESULTS\RNN_1\evaluation_metrics.xlsx")
df4_1 = pd.read_excel(r"C:\Users\GY\Desktop\xlw\models\LSTM\RESULTS\LSTM_1\evaluation_metrics.xlsx")
TimeFusionNet_values_array_1 = df_1.loc[0:4, 'Value'].values
CNNLSTM_values_array_1 = df1_1.loc[0:4, 'Value'].values
GRU_values_array_1 = df2_1.loc[0:4, 'Value'].values
RNN_values_array_1 = df3_1.loc[0:4, 'Value'].values
LSTM_values_array_1 = df4_1.loc[0:4, 'Value'].values


#第三步预测指标
df_2 = pd.read_excel(r"C:\Users\GY\Desktop\xlw\pic\20250420_145551 lag=120\工作簿2.xlsx")
df1_2 = pd.read_excel(r"C:\Users\GY\Desktop\xlw\models\CNN-LSTM\RESULTS\CNNLSTM_2\evaluation_metrics.xlsx")
df2_2 = pd.read_excel(r"C:\Users\GY\Desktop\xlw\models\GRU\RESULTS\GRU_2\evaluation_metrics.xlsx")
df3_2 = pd.read_excel(r"C:\Users\GY\Desktop\xlw\models\RNN\RESULTS\RNN_2\evaluation_metrics.xlsx")
df4_2 = pd.read_excel(r"C:\Users\GY\Desktop\xlw\models\LSTM\RESULTS\LSTM_2\evaluation_metrics.xlsx")
TimeFusionNet_values_array_2 = df_2.loc[0:4, 'Value'].values
CNNLSTM_values_array_2 = df1_2.loc[0:4, 'Value'].values
GRU_values_array_2 = df2_2.loc[0:4, 'Value'].values
RNN_values_array_2 = df3_2.loc[0:4, 'Value'].values
LSTM_values_array_2 = df4_2.loc[0:4, 'Value'].values

#第四步预测指标
df_3 = pd.read_excel(r"C:\Users\GY\Desktop\xlw\pic\20250420_215215 lag=180\工作簿3.xlsx")
df1_3 = pd.read_excel(r"C:\Users\GY\Desktop\xlw\models\CNN-LSTM\RESULTS\CNNLSTM_3\evaluation_metrics.xlsx")
df2_3 = pd.read_excel(r"C:\Users\GY\Desktop\xlw\models\GRU\RESULTS\GRU_3\evaluation_metrics.xlsx")
df3_3 = pd.read_excel(r"C:\Users\GY\Desktop\xlw\models\RNN\RESULTS\RNN_3\evaluation_metrics.xlsx")
df4_3 = pd.read_excel(r"C:\Users\GY\Desktop\xlw\models\LSTM\RESULTS\lstm_3\工作簿1.xlsx")
TimeFusionNet_values_array_3 = df_3.loc[0:4, 'Value'].values
CNNLSTM_values_array_3 = df_3.loc[0:4, 'Value'].values
GRU_values_array_3 = df_3.loc[0:4, 'Value'].values
RNN_values_array_3 = df_3.loc[0:4, 'Value'].values
LSTM_values_array_3 = df_3.loc[0:4, 'Value'].values


def plot_improvement_rates(baseline_model, model2, model3, model4, model5, model_names=None, save_path=None):
    """
    绘制并保存四个对比模型相对于基线模型的MAE、MSE和MAPE改进率(IR)柱状图
    """
    # 设置中文字体
    plt.rcParams["font.family"] = ["SimHei", "SimHei", "SimHei"]

    # 默认模型名称
    if model_names is None:
        model_names = ["TimeF   usionNet", 'CNN_LSTM', "GRU", "RNN", "LSTM"]

    # 定义每个模型对应的颜色
    model_colors = {
        "CNNLSTM": "#2EF7FF",
        "GRU": "#312FFF",
        "RNN": "#9831FF",
        "LSTM": "#FF8D0A"
    }

    # 提取数据
    baseline_metrics = baseline_model
    compare_models = [model2, model3, model4, model5]
    compare_names = model_names[1:]

    # 确保颜色顺序与模型顺序一致
    colors = [model_colors.get(name, "#999999") for name in compare_names]

    # 计算改进率(IR)
    def calculate_ir(baseline, compare):
        return (abs(baseline - compare) / baseline) * 100

    mae_ir = [calculate_ir(baseline_metrics[0], model[0]) for model in compare_models]
    mse_ir = [calculate_ir(baseline_metrics[1], model[1]) for model in compare_models]
    mape_ir = [calculate_ir(baseline_metrics[3], model[3]) for model in compare_models]

    # 绘图
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    bar_width = 0.8
    index = np.arange(len(compare_models))

    # MAE IR柱状图 - 为每个柱子设置不同颜色
    for i in range(len(compare_models)):
        axes[0].bar(index[i], mae_ir[i], bar_width, color=colors[i])
    axes[0].set_title('MAE improvement rate(IR)', fontsize=18)  # 标题18
    axes[0].set_ylabel('improvement rate (%)', fontsize=18)  # y轴名18
    axes[0].set_xticks(index)
    axes[0].set_xticklabels(compare_names, rotation=45, ha='right', fontsize=18)  # x刻度文字18
    axes[0].tick_params(axis='y', labelsize=18)  # y刻度文字18
    axes[0].axhline(y=0, color='black', linestyle='-', alpha=0.3)
    for i, v in enumerate(mae_ir):
        axes[0].text(i, v + 0.5 if v > 0 else v - 0.5, f'{v:.2f}%', ha='center', va='bottom' if v > 0 else 'top',
                     fontsize=16)

    # MSE IR柱状图 - 为每个柱子设置不同颜色
    for i in range(len(compare_models)):
        axes[1].bar(index[i], mse_ir[i], bar_width, color=colors[i])
    axes[1].set_title('MSE improvement rate(IR)', fontsize=18)  # 标题18
    axes[1].set_ylabel('improvement rate (%)', fontsize=18)  # y轴名18
    axes[1].set_xticks(index)
    axes[1].set_xticklabels(compare_names, rotation=45, ha='right', fontsize=18)  # x刻度文字18
    axes[1].tick_params(axis='y', labelsize=18)  # y刻度文字18
    axes[1].axhline(y=0, color='black', linestyle='-', alpha=0.3)
    for i, v in enumerate(mse_ir):
        axes[1].text(i, v + 0.5 if v > 0 else v - 0.5, f'{v:.2f}%', ha='center', va='bottom' if v > 0 else 'top',
                     fontsize=16)

    # MAPE IR柱状图 - 为每个柱子设置不同颜色
    for i in range(len(compare_models)):
        axes[2].bar(index[i], mape_ir[i], bar_width, color=colors[i])
    axes[2].set_title('MAPE improvement rate(IR)', fontsize=18)  # 标题18
    axes[2].set_ylabel('improvement rate (%)', fontsize=18)  # y轴名18
    axes[2].set_xticks(index)
    axes[2].set_xticklabels(compare_names, rotation=45, ha='right', fontsize=18)  # x刻度文字18
    axes[2].tick_params(axis='y', labelsize=18)  # y刻度文字18
    axes[2].axhline(y=0, color='black', linestyle='-', alpha=0.3)
    for i, v in enumerate(mape_ir):
        axes[2].text(i, v + 0.5 if v > 0 else v - 0.5, f'{v:.2f}%', ha='center', va='bottom' if v > 0 else 'top',
                     fontsize=16)

    plt.tight_layout()
    # 保存图表
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"图表已保存至：{save_path}")

    return fig, axes


plot_improvement_rates(TimeFusionNet_values_array_0, CNNLSTM_values_array_0, GRU_values_array_0, RNN_values_array_0, LSTM_values_array_0, ["TimeFusionNet", "CNNLSTM", "GRU", "RNN", "LSTM"], "C:/Users/GY/Desktop/xlw/pic/zhuzhuangtu/0")
plot_improvement_rates(TimeFusionNet_values_array_1, CNNLSTM_values_array_1, GRU_values_array_1, RNN_values_array_1, LSTM_values_array_1, ["TimeFusionNet", "CNNLSTM", "GRU", "RNN", "LSTM"], "C:/Users/GY/Desktop/xlw/pic/zhuzhuangtu/1")
plot_improvement_rates(TimeFusionNet_values_array_2, CNNLSTM_values_array_2, GRU_values_array_2, RNN_values_array_2, LSTM_values_array_2, ["TimeFusionNet", "CNNLSTM", "GRU", "RNN", "LSTM"], "C:/Users/GY/Desktop/xlw/pic/zhuzhuangtu/2")
plot_improvement_rates(TimeFusionNet_values_array_3, CNNLSTM_values_array_3, GRU_values_array_3, RNN_values_array_3, LSTM_values_array_3, ["TimeFusionNet", "CNNLSTM", "GRU", "RNN", "LSTM"], "C:/Users/GY/Desktop/xlw/pic/zhuzhuangtu/3")

