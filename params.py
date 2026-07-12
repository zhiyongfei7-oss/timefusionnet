class Config:
    # 数据参数
    data_path = "C:/Users/GY/Desktop/xlw/triple1.xlsx"
    window_size = 120  # 60
    step = 10
    lag = 60
    output_dim = 1

    # 模型参数
    num_heads = 4
    attention_dim = 32

    # 训练参数
    batch_size = 64
    lr = 0.007
    epochs = 100

    # 地址


import os
import shutil


def clear_directory(directory_path):
    """
    清空指定目录下的所有文件和子目录，但保留目录本身
    :param directory_path: 目录路径
    """
    if not os.path.exists(directory_path):
        print(f"目录 {directory_path} 不存在，无需清空")
        return

    for root, dirs, files in os.walk(directory_path, topdown=False):
        # 删除所有文件
        for file_name in files:
            file_path = os.path.join(root, file_name)
            os.remove(file_path)
            print(f"删除文件: {file_path}")

        # 删除所有子目录
        for dir_name in dirs:
            dir_path = os.path.join(root, dir_name)
            shutil.rmtree(dir_path)
            print(f"删除目录: {dir_path}")


config = Config()
