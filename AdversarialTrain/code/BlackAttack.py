# 此文件加载已经训练好的白盒模型，并依据预测输出和梯度输出，生成对抗样本并存储于data.npy文件中
# 此模型使用的原始总样本来自于黑盒模型筛选出的能够被黑盒模型正确预测的模型
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import argparse
import matplotlib.pylab as plt
import numpy as np
from WhiteAttackModel import Net as WhiteNet
from model import Net as BlackNet

WhiteNet = WhiteNet()
WhiteNet.load_state_dict(torch.load("white_model.pt"))

BlackNet = BlackNet()
BlackNet.load_state_dict(torch.load("fashion_mnist_cnn.pt"))
epsilons = .3


def fgsm_attack(image, epsilon, data_grad):
    # Collect the element-wise sign of the data gradient
    sign_data_grad = data_grad.sign()
    # Create the perturbed image by adjusting each pixel of the input image
    perturbed_image = image + epsilon * sign_data_grad
    # Adding clipping to maintain [0,1] range
    # perturbed_image = torch.clamp(perturbed_image, 0, 255)# Normalize之后数据变成均值为0方差为1 有正有负 不应该clamp
    # Return the perturbed image
    # return perturbed_image
    return perturbed_image


def test(device, test_loader, epsilon):
    Sum = 0  # 总样本数=8989
    Success = 0  # 依照要求成功完成攻击任务的样本数  (init_pred +1)%10 = final_pred
    adv_examples = []

    for data, target in test_loader:
        # target等于在黑盒上预测的结果
        data, target = torch.tensor(data), torch.tensor(target)
        # 使用数据在白盒模型上产生的梯度 对数据进行扰动
        data = data.view(1, 1, 28, 28)
        target = target.view(1)
        data, target = data.to(device), target.to(device)
        data.requires_grad = True
        # print(target.shape)
        WhiteOutput = WhiteNet(data)
        loss = F.nll_loss(WhiteOutput, target)
        Sum += 1
        # Zero all existing gradients
        WhiteNet.zero_grad()
        # Calculate gradients of model in backward pass
        loss.backward()
        # Collect datagrad
        data_grad = data.grad.data
        # Call FGSM Attack
        perturbed_data = fgsm_attack(data, epsilon, data_grad)
        PerturbedBlackOutput = BlackNet(perturbed_data)
        final_pred = PerturbedBlackOutput.max(1, keepdim=True)[1]  # get the index of the max log-probability
        if (target + 1) % 10 == final_pred.item():
            Success += 1
            pre_ex = data.squeeze().detach().cpu().numpy()
            adv_ex = perturbed_data.squeeze().detach().cpu().numpy()
            adv_examples.append((target.item(), pre_ex, final_pred.item(), adv_ex))
            print("Attack succeed for sample {}".format(Sum))
        else:
            print("Attack fails for sample {}".format(Sum))
    Success_rate = Success / Sum
    print("Epsilon: {}\tSuccess_rate = {} / {} = {}".format(epsilon, Success, Sum, Success_rate))
    return Success_rate, adv_examples
    # Return the Success_rate and an adversarial example


parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
parser.add_argument('--attack-batch-size', type=int, default=1, metavar='N',
                    help='input batch size for testing (default: 1000)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
args = parser.parse_args()
test_loader = np.load('filter_data.npy')
test_loader = test_loader.tolist()
use_cuda = not args.no_cuda and torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
acc, ex = test(device, test_loader, epsilons)
print("Saving List in data.npy")
examples = np.array(ex)
np.save('data.npy', examples)
# Epsilon: 0.3	Success_rate = 138 / 8989 = 0.015352097007453554