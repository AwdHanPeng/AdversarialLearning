import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import argparse
import matplotlib.pylab as plt
import numpy as np
from model import Net

# 此文件加载已经训练好的白盒模型，并依据预测输出和梯度输出，生成对抗样本并存储于data.npy文件中
model = Net()
model.load_state_dict(torch.load("fashion_mnist_cnn.pt"))
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


def test(model, device, test_loader, epsilon):
    Sum = 0  # 未受攻击前可以被正确分类的样本总数
    # Accuracy counter
    Success = 0  # 依照要求成功完成攻击任务的样本数  (init_pred +1)%10 = final_pred
    adv_examples = []
    count = 1  # 总样本计数
    # Loop over all examples in test set
    for data, target in test_loader:

        # Send the data and label to the device
        data, target = data.to(device), target.to(device)

        # Set requires_grad attribute of tensor. Important for Attack
        data.requires_grad = True

        # Forward pass the data through the model
        output = model(data)
        init_pred = output.max(1, keepdim=True)[1]  # get the index of the max log-probability

        # If the initial prediction is wrong, dont bother attacking, just move on
        if init_pred.item() != target.item():
            print("Sample {} is a improper sample".format(count))  # 只有初始分类正确的样本才有攻击的必要
            count += 1
            continue
        Sum += 1
        # Calculate the loss
        loss = F.nll_loss(output, target)

        # Zero all existing gradients
        model.zero_grad()

        # Calculate gradients of model in backward pass
        loss.backward()

        # Collect datagrad
        data_grad = data.grad.data

        # Call FGSM Attack
        perturbed_data = fgsm_attack(data, epsilon, data_grad)

        # Re-classify the perturbed image
        output = model(perturbed_data)
        final_pred = output.max(1, keepdim=True)[1]  # get the index of the max log-probability
        if (init_pred.item() + 1) % 10 == final_pred.item():
            Success += 1
            pre_ex = data.squeeze().detach().cpu().numpy()
            adv_ex = perturbed_data.squeeze().detach().cpu().numpy()
            adv_examples.append((init_pred.item(), pre_ex, final_pred.item(), adv_ex))
            print("Attack succeed for sample {}".format(count))
            count += 1
        else:
            print("Attack fails for sample {}".format(count))
            count += 1
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
test_loader = torch.utils.data.DataLoader(
    datasets.FashionMNIST('../data', train=False, transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])),
    batch_size=args.attack_batch_size, shuffle=True)
use_cuda = not args.no_cuda and torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
acc, ex = test(model, device, test_loader, epsilons)
print("Saving List in data.npy")
examples = np.array(ex)
np.save('data.npy', examples)
# Epsilon: 0.3	Success_rate = 277 / 8989 = 0.030815441094671266