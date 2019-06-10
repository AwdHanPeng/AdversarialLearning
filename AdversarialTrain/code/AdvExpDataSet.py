import numpy as np
from torch.utils.data import Dataset


# 对抗生成样本加上原始训练集样本共61652个 构建出用于训练的dataset
# print(len(ex)) 61652
class AdvExpDataset(Dataset):
    def __init__(self, root, transform=None):
        self.data = np.load(root).tolist()
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data, target = self.data[idx]
        if self.transform is not None:
            data = self.transform(data)
        return data, target
