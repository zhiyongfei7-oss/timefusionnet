import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import os
import pywt
from datetime import datetime

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from torch import nn
from config.params import config, clear_directory
from data.datasets import SlidingWindowDataset
from models.multi_branch import MultiBranchModel
from models.spatiotemporal import SpatioTemporalNetwork
from utils.trainer import Trainer
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

# 设备配置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# 初始化组件
dataset = SlidingWindowDataset(config.data_path,
                               config.window_size,
                               config.step,
                               config.lag)
# 将数据集划分为训练集和测试集
train_data, test_data = train_test_split(dataset, test_size=0.2, random_state=42)

train_dataloader = DataLoader(train_data, batch_size=config.batch_size, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=config.batch_size, shuffle=False)

main_model = MultiBranchModel(config)

output_dim = config.output_dim
st_network = SpatioTemporalNetwork(output_dim)


# 组合模型
class FullModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.multi_branch = MultiBranchModel(config)
        self.st_network = SpatioTemporalNetwork(output_dim)
        self.to(device)

    def forward(self, x):
        x = self.multi_branch(x)
        return self.st_network(x)


def calculate_metrics(y_true, y_pred):
    y_true = y_true.ravel()
    y_pred = y_pred.ravel()
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    epsilon = 1e-8  # 防止除以零
    mape = np.mean(np.abs((y_true - y_pred) / (y_true + epsilon))) * 100
    r2 = r2_score(y_true, y_pred)
    return mae, mse, mape, r2


# 绘制不同范围的图表
def plot_and_save(targets, predictions, title, filename, start_idx, end_idx, time_folder):
    plt.figure(figsize=(12, 6))
    plt.plot(targets[start_idx:end_idx], label='True Values')
    plt.plot(predictions[start_idx:end_idx], label='Predictions')
    plt.title(title)
    plt.xlabel('Sample')
    plt.ylabel('Value')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(time_folder, filename))
    plt.close()


# 绘制散点图
def plot_scatter(true, pred, title, filename, time_folder):
    plt.figure(figsize=(8, 6))
    plt.scatter(true, pred, alpha=0.5, label='Predictions')
    plt.plot([true.min(), true.max()], [true.min(), true.max()], 'r--', label='Ideal')
    plt.xlabel('True Values')
    plt.ylabel('Predictions')
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(time_folder, filename))
    plt.close()


def train_model():
    # 训练流程
    trainer = Trainer(FullModel(), config)

    # 训练循环
    train_losses = []
    test_losses = []

    # 初始化变量来保存最佳模型和对应的test loss
    best_loss = float('inf')
    saved_models = []

    # 创建保存目录
    save_dir = r"C:\Users\GY\Desktop\xlw\pic"
    # save_dir = r"C:\Users\GY\Desktop\xlw\pic_list1"
    # save_dir = r"C:\Users\GY\Desktop\xlw\pic_list2"
    os.makedirs(save_dir, exist_ok=True)

    # 获取当前时间
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")

    # 创建以日期小时分钟秒命名的子文件夹
    time_folder = os.path.join(save_dir, current_time)
    os.makedirs(time_folder, exist_ok=True)
    model_folder = os.path.join(time_folder, "models")
    model_path = os.path.join(model_folder, "best_model.pth")  # 指定文件名
    os.makedirs(model_folder, exist_ok=True)

    for epoch in range(config.epochs):
        train_loss = trainer.train_epoch(train_dataloader)
        test_loss = trainer.evaluate(test_dataloader)
        train_losses.append(train_loss)
        test_losses.append(test_loss)
        print(f"Epoch {epoch + 1}, Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}")

        # 检查当前模型的test loss是否是最后10次epoch中最低的
        if test_loss < best_loss:
            if model_folder is not None:
                clear_directory(model_folder)
            best_loss = test_loss
            # 保存当前最佳模型
            trainer.save_model(model_path)
            print(f"Saved model with best test loss: {best_loss:.4f}")

    # 在训练后对测试集进行预测
    train_predictions, train_targets = trainer.predict(train_dataloader)
    test_predictions, test_targets = trainer.predict(test_dataloader)

    # 展平数据确保维度一致
    train_true = train_targets.ravel()
    train_pred = train_predictions.ravel()
    test_true = test_targets.ravel()
    test_pred = test_predictions.ravel()

    # 创建带时间戳的文件名
    excel_filename = f"predictions_{current_time}"
    train_excel_path = os.path.join(time_folder, f"{excel_filename}_train.xlsx")
    test_excel_path = os.path.join(time_folder, f"{excel_filename}_test.xlsx")

    train_true = train_targets.reshape(-1)

    # 创建DataFrame并保存
    pd.DataFrame({
        'True Values': train_true,
        'Predicted Values': train_pred
    }).to_excel(train_excel_path, index=False, engine='openpyxl')

    pd.DataFrame({
        'True Values': test_true,
        'Predicted Values': test_pred
    }).to_excel(test_excel_path, index=False, engine='openpyxl')

    # 计算并打印指标
    train_mae, train_mse, train_mape, train_r2 = calculate_metrics(train_targets, train_predictions)
    print(f"Training Metrics:")
    print(f"MAE: {train_mae:.4f}, MSE: {train_mse:.4f}, MAPE: {train_mape:.2f}%, R²: {train_r2:.4f}")

    test_mae, test_mse, test_mape, test_r2 = calculate_metrics(test_targets, test_predictions)
    print(f"Test Metrics:")
    print(f"MAE: {test_mae:.4f}, MSE: {test_mse:.4f}, MAPE: {test_mape:.2f}%, R²: {test_r2:.4f}")

    # 大范围作图
    plot_and_save(
        train_targets,
        train_predictions,
        'Training Set - Full Range',
        f"{current_time}_大.png",
        0, 3000,
        time_folder
    )

    plot_and_save(
        test_targets,
        test_predictions,
        'Test Set - Full Range',
        f"{current_time}_大.png",
        0, len(test_targets),
        time_folder
    )

    # 中范围作图
    plot_and_save(
        train_targets,
        train_predictions,
        'Training Set - Medium Range',
        f"{current_time}_中.png",
        100, 250,
        time_folder
    )

    plot_and_save(
        test_targets,
        test_predictions,
        'Test Set - Medium Range',
        f"{current_time}_中.png",
        100, 250,
        time_folder
    )

    # 小范围作图
    plot_and_save(
        train_targets,
        train_predictions,
        'Training Set - Small Range',
        f"{current_time}_小.png",
        200, 240,
        time_folder
    )

    plot_and_save(
        test_targets,
        test_predictions,
        'Test Set - Small Range',
        f"{current_time}_小.png",
        200, 240,
        time_folder
    )

    plot_and_save(
        test_targets,
        test_predictions,
        'Test Set - Small Range',
        f"{current_time}_小1.png",
        85, 125,
        time_folder
    )

    plot_scatter(train_targets, train_predictions, 'Training Set Predictions', f"{current_time}_train_scatter.png",
                 time_folder)
    plot_scatter(test_targets, test_predictions, 'Test Set Predictions', f"{current_time}_test_scatter.png",
                 time_folder)

    # 绘制训练和测试损失曲线
    plt.figure(figsize=(12, 6))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(test_losses, label='Test Loss')
    plt.title('Training and Test Loss Curve')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(time_folder, 'loss_curve.png'))
    plt.close()

    # 写入指标到文件
    metrics_path = os.path.join(time_folder, "evaluation_metrics.txt")
    with open(metrics_path, 'w', encoding='utf-8') as f:
        f.write(f"=== 模型评估指标 ({current_time}) ===\n\n")
        f.write("[Training Set]\n")
        f.write(f"MAE (平均绝对误差): {train_mae:.4f}\n")
        f.write(f"MSE (均方误差): {train_mse:.4f}\n")
        f.write(f"MAPE (平均百分比误差): {train_mape:.2f}%\n")
        f.write(f"R² (决定系数): {train_r2:.4f}\n\n")

        f.write("[Test Set]\n")
        f.write(f"MAE (平均绝对误差): {test_mae:.4f}\n")
        f.write(f"MSE (均方误差): {test_mse:.4f}\n")
        f.write(f"MAPE (平均百分比误差): {test_mape:.2f}%\n")
        f.write(f"R² (决定系数): {test_r2:.4f}\n")

    #写入指标到xlsx文件
    metrics_df = pd.DataFrame({
        'Metric': ['MAE', 'MSE', 'RMSE', 'MAPE', 'R²'],
        'Value': [test_mae, test_mse, test_mape, test_r2]
    })
    metrics_path = os.path.join(time_folder, "evaluation_metrics.xlsx")
    metrics_df.to_excel(metrics_path, index=False, engine='openpyxl')
    print(f"保存评估指标到: {metrics_path}")

    print(f"所有图表已保存到 {save_dir}")


if __name__ == "__main__":
    train_model()
