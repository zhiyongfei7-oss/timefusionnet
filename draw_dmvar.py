import pandas as pd
import numpy as np
from pathlib import Path

# 四组预测结果的文件路径
file_paths_list = [
    [
        r"C:\Users\GY\Desktop\xlw\pic\20250705_085307_最好\predictions_20250705_085307_test.xlsx",
        r"C:\Users\GY\Desktop\xlw\models\CNN-LSTM\RESULTS\CNNLSTM_0\test_predictions_original.xlsx",
        r"C:\Users\GY\Desktop\xlw\models\GRU\RESULTS\GRU_0\test_predictions_original.xlsx",
        r"C:\Users\GY\Desktop\xlw\models\LSTM\RESULTS\LSTM_0\test_predictions_original.xlsx",
        r"C:\Users\GY\Desktop\xlw\models\RNN\RESULTS\RNN_0\test_predictions_original.xlsx"
    ],
    [
        r"C:\Users\GY\Desktop\xlw\pic\20250419_220806 lag=60\predictions_20250419_220806_test.xlsx",
        r"C:\Users\GY\Desktop\xlw\models\CNN-LSTM\RESULTS\CNNLSTM_1\test_predictions_original.xlsx",
        r"C:\Users\GY\Desktop\xlw\models\GRU\RESULTS\GRU_1\test_predictions_original.xlsx",
        r"C:\Users\GY\Desktop\xlw\models\LSTM\RESULTS\LSTM_1\test_predictions_original.xlsx",
        r"C:\Users\GY\Desktop\xlw\models\RNN\RESULTS\RNN_1\test_predictions_original.xlsx"
    ],
    [
        r"C:\Users\GY\Desktop\xlw\pic\20250420_145551 lag=120\predictions_20250420_145551_test.xlsx",
        r"C:\Users\GY\Desktop\xlw\models\CNN-LSTM\RESULTS\CNNLSTM_2\test_predictions_original.xlsx",
        r"C:\Users\GY\Desktop\xlw\models\GRU\RESULTS\GRU_2\test_predictions_original.xlsx",
        r"C:\Users\GY\Desktop\xlw\models\LSTM\RESULTS\LSTM_2\test_predictions_original.xlsx",
        r"C:\Users\GY\Desktop\xlw\models\RNN\RESULTS\RNN_2\test_predictions_original.xlsx"
    ],
    [
        r"C:\Users\GY\Desktop\xlw\pic\20250420_215215 lag=180\predictions_20250420_215215_test.xlsx",
        r"C:\Users\GY\Desktop\xlw\models\CNN-LSTM\RESULTS\CNNLSTM_3\test_predictions_original.xlsx",
        r"C:\Users\GY\Desktop\xlw\models\GRU\RESULTS\GRU_3\test_predictions_original.xlsx",
        r"C:\Users\GY\Desktop\xlw\models\LSTM\RESULTS\lstm_3\工作簿1.xlsx",
        r"C:\Users\GY\Desktop\xlw\models\RNN\RESULTS\RNN_3\test_predictions_original.xlsx"
    ]
]

# 输出文件夹路径
output_dir = Path(r"C:\Users\GY\Desktop\xlw\pic\dm.var")
output_dir.mkdir(parents=True, exist_ok=True)

# 模型名称列表
model_names = ["Your Model", "CNN-LSTM", "GRU", "LSTM", "RNN"]

# 预测结果组名称
group_names = ["Lag=60", "Lag=120", "Lag=180", "Lag=300"]


# 定义损失函数，这里使用MSE损失
def loss_function(y_true, y_pred):
    return (y_true - y_pred) ** 2


# 计算DM统计量
def calculate_dm(model_errors, rival_errors, valid_true_values):
    # 计算预测值
    model_pred = valid_true_values - model_errors
    rival_pred = valid_true_values - rival_errors

    # 计算损失函数差值
    loss_diff = loss_function(valid_true_values, model_pred) - loss_function(valid_true_values, rival_pred)

    # 计算均值和方差
    mean_diff = np.mean(loss_diff)
    var_diff = np.var(loss_diff, ddof=1)  # 使用样本方差

    # 计算DM统计量
    if var_diff == 0:
        return float('inf') if mean_diff > 0 else float('-inf')
    dm = mean_diff / np.sqrt(var_diff / len(loss_diff))
    return dm


# 读取数据并处理长度不一致问题
def read_data(file_path, expected_length=None):
    df = pd.read_excel(file_path, skiprows=1, usecols=[0, 1])

    # 获取真实值和预测值
    true_values = df.iloc[:, 0].values
    pred_values = df.iloc[:, 1].values

    # 检查数据长度
    if expected_length is not None:
        if len(true_values) > expected_length:
            print(f"警告: 文件 {file_path} 的真实值长度 ({len(true_values)}) 超过预期长度 ({expected_length})，将截断")
            true_values = true_values[:expected_length]
        elif len(true_values) < expected_length:
            print(f"警告: 文件 {file_path} 的真实值长度 ({len(true_values)}) 小于预期长度 ({expected_length})，将填充NaN")
            true_values = np.pad(true_values, (0, expected_length - len(true_values)), 'constant',
                                 constant_values=np.nan)

        if len(pred_values) > expected_length:
            print(f"警告: 文件 {file_path} 的预测值长度 ({len(pred_values)}) 超过预期长度 ({expected_length})，将截断")
            pred_values = pred_values[:expected_length]
        elif len(pred_values) < expected_length:
            print(f"警告: 文件 {file_path} 的预测值长度 ({len(pred_values)}) 小于预期长度 ({expected_length})，将填充NaN")
            pred_values = np.pad(pred_values, (0, expected_length - len(pred_values)), 'constant',
                                 constant_values=np.nan)

    return true_values, pred_values


# 为每组预测结果创建一个Excel文件
for group_idx, file_paths in enumerate(file_paths_list):
    print(f"\n处理第 {group_idx + 1} 组预测结果: {group_names[group_idx]}")

    # 读取第一个文件以确定基准长度
    base_true, base_pred = read_data(file_paths[0])
    base_length = len(base_true)

    # 存储各模型预测值和误差
    predictions = []
    errors = []

    # 基准真实值
    true_values = base_true

    for i, path in enumerate(file_paths):
        # 读取数据并对齐长度
        true, pred = read_data(path, base_length)

        # 计算误差，处理NaN值
        error = true - pred
        errors.append(error)

    # 以第一个模型为基准，与其他模型对比计算DM值
    base_model_errors = errors[0]
    dm_values = []

    for i in range(1, len(errors)):
        rival_errors = errors[i]

        # 找出两个误差数组中都有效的索引
        valid_mask = ~np.isnan(base_model_errors) & ~np.isnan(rival_errors)
        valid_base_errors = base_model_errors[valid_mask]
        valid_rival_errors = rival_errors[valid_mask]
        valid_true = true_values[valid_mask]  # 使用相同的有效索引选择真实值

        # 确保有效数据长度一致
        print(
            f"基准模型有效数据长度: {len(valid_base_errors)}, 对比模型有效数据长度: {len(valid_rival_errors)}, 真实值有效数据长度: {len(valid_true)}")

        # 计算DM值
        dm_value = calculate_dm(valid_base_errors, valid_rival_errors, valid_true)
        dm_values.append(dm_value)
        print(f"DM值 ({model_names[0]} vs {model_names[i]}): {dm_value:.4f}")

    # 计算每个模型预测误差的VAR值
    var_values = []
    for i, err in enumerate(errors):
        # 处理NaN值
        valid_err = err[~np.isnan(err)]
        var_value = np.var(valid_err, ddof=1)  # 使用样本方差
        var_values.append(var_value)
        print(f"VAR值 ({model_names[i]}): {var_value:.4f}")

    # 创建DataFrame并保存到Excel
    # DM值结果
    dm_df = pd.DataFrame({
        "对比模型": model_names[1:],
        f"DM值 (vs {model_names[0]})": dm_values
    })

    # VAR值结果
    var_df = pd.DataFrame({
        "模型": model_names,
        "VAR值": var_values
    })

    # 保存到Excel文件
    group_output_dir = output_dir / f"Group_{group_idx + 1}_{group_names[group_idx]}"
    group_output_dir.mkdir(parents=True, exist_ok=True)

    dm_file = group_output_dir / f"dm_values_{group_names[group_idx]}.xlsx"
    var_file = group_output_dir / f"var_values_{group_names[group_idx]}.xlsx"

    dm_df.to_excel(dm_file, index=False)
    var_df.to_excel(var_file, index=False)

    print(f"DM值已保存到: {dm_file}")
    print(f"VAR值已保存到: {var_file}")

# 创建汇总表
summary_dm_data = []
summary_var_data = []

for group_idx, file_paths in enumerate(file_paths_list):
    group_name = group_names[group_idx]

    # 读取第一个文件以确定基准长度
    base_true, _ = read_data(file_paths[0])
    base_length = len(base_true)

    # 计算VAR值
    var_values = []
    for i, path in enumerate(file_paths):
        true, pred = read_data(path, base_length)
        error = true - pred

        # 处理NaN值
        valid_error = error[~np.isnan(error)]
        var_value = np.var(valid_error, ddof=1)
        var_values.append(var_value)

    # 计算DM值（只计算第一个模型与其他模型的比较）
    base_true, base_pred = read_data(file_paths[0], base_length)
    base_error = base_true - base_pred

    dm_values = []
    for i in range(1, len(file_paths)):
        rival_true, rival_pred = read_data(file_paths[i], base_length)
        rival_error = rival_true - rival_pred

        # 找出两个误差数组中都有效的索引
        valid_mask = ~np.isnan(base_error) & ~np.isnan(rival_error)
        valid_base_error = base_error[valid_mask]
        valid_rival_error = rival_error[valid_mask]
        valid_true = base_true[valid_mask]  # 使用相同的有效索引选择真实值

        dm_value = calculate_dm(valid_base_error, valid_rival_error, valid_true)
        dm_values.append(dm_value)

    # 添加到汇总数据
    for i, model_name in enumerate(model_names):
        if i == 0:  # 只添加一次基准模型的VAR
            summary_var_data.append({
                "Group": group_name,
                "Model": model_name,
                "VAR": var_values[i]
            })
        elif i <= len(dm_values):  # 添加对比模型的DM和VAR
            summary_dm_data.append({
                "Group": group_name,
                "Base Model": model_names[0],
                "Rival Model": model_name,
                "DM Value": dm_values[i - 1]
            })
            summary_var_data.append({
                "Group": group_name,
                "Model": model_name,
                "VAR": var_values[i]
            })

# 创建汇总DataFrame
summary_dm_df = pd.DataFrame(summary_dm_data)
summary_var_df = pd.DataFrame(summary_var_data)

# 保存汇总表
summary_dm_file = output_dir / "summary_dm_values.xlsx"
summary_var_file = output_dir / "summary_var_values.xlsx"

summary_dm_df.to_excel(summary_dm_file, index=False)
summary_var_df.to_excel(summary_var_file, index=False)

print(f"\n汇总DM值已保存到: {summary_dm_file}")
print(f"汇总VAR值已保存到: {summary_var_file}")