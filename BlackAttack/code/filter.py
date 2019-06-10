# 此文件用以从测试数据集中筛选出需要进行攻击的样本
# 使用的模型为待攻击的黑盒模型
import torch
import torch.nn.functional as F
from torchvision import datasets, transforms
import argparse
from model import Net as BlackNet
import numpy as np

BlackNet = BlackNet()
BlackNet.load_state_dict(torch.load("fashion_mnist_cnn.pt"))
filter_data = []


def filter(args, model, device, test_loader):
    model.eval()
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            if pred.eq(target.view_as(pred)):
                filter_data.append((data.squeeze().detach().cpu().numpy(), target.item()))


parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
parser.add_argument('--test-batch-size', type=int, default=1, metavar='N',
                    help='input batch size for testing (default: 1)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
args = parser.parse_args()
test_loader = torch.utils.data.DataLoader(
    datasets.FashionMNIST('../data', train=False, transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])),
    batch_size=args.test_batch_size, shuffle=True)
use_cuda = not args.no_cuda and torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
print("开始筛选----------------------------------")
filter(args, BlackNet, device, test_loader)
print("筛选完毕----------------------------------")
print("得到{}个正确预测的样本".format(len(filter_data)))
filter_data = np.array(filter_data)
np.save('filter_data.npy', filter_data)
