import torch
from torchvision import datasets, transforms
import numpy as np

# 将对抗样本掺入训练数据集中 新的数据集存储与new_data.npy中
train_loader = torch.utils.data.DataLoader(
    datasets.FashionMNIST('../data', train=True, download=True,
                          transform=transforms.Compose([
                              transforms.ToTensor(),
                              transforms.Normalize((0.1307,), (0.3081,))
                          ])),
    batch_size=1, shuffle=True)
adv_data = np.load('data.npy').tolist()
new_data = []
for i, (init_pred, pre_ex, final_pred, adv_ex) in enumerate(adv_data):
    new_data.append((adv_ex, final_pred))

for data, target in train_loader:
    data = data.squeeze().detach().cpu().numpy()
    target = target.item()
    new_data.append((data, target))

new_data = np.array(new_data)
print("Saving List in new_data.npy")
np.save('new_data.npy', new_data)
print("Complete")
# new_data中包含原始训练数据集和对抗生成的样本
