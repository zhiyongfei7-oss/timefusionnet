import numpy as np
import torch
from torch.utils.data import Dataset
import pandas as pd


class SlidingWindowDataset(Dataset):
    def __init__(self, data_path, window_size, step, lag):
        df = pd.read_excel(data_path, engine='openpyxl', sheet_name="f_cao")
        # 将 DataFrame 转换为 NumPy 数组
        raw_data = df.values

        self.features = []
        self.labels = []
        for hour_data in raw_data:
            self.features.append(hour_data[:600].reshape(10, 60).T)
            self.labels.append(hour_data[600])

        self.full_sequence = np.concatenate(self.features, axis=0)
        self.labels = np.array(self.labels)

        self._create_windows(window_size, step, lag)

    def _create_windows(self, window_size, step, lag):
        self.windows = []
        self.targets = []
        total_minutes = len(self.full_sequence)

        for start in range(0, total_minutes - window_size + 1, step):
            end = start + window_size
            label_hour = (end + lag) // 60
            if label_hour < len(self.labels):
                self.windows.append((start, end))
                self.targets.append(label_hour)

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        start, end = self.windows[idx]
        features = self.full_sequence[start:end]
        label = self.labels[self.targets[idx]]
        return torch.FloatTensor(features), torch.FloatTensor([label])
