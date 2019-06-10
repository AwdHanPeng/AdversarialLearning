# 在新分类器上重复白盒攻击 且白盒攻击的使用样本为测试集
import torch
import torch.nn.functional as F
from torchvision import datasets, transforms
import argparse
import numpy as np
from model import Net

model = Net()
model.load_state_dict(torch.load("new_fashion_mnist_cnn.pt"))
epsilons = .3


def fgsm_attack(image, epsilon, data_grad):
    sign_data_grad = data_grad.sign()
    perturbed_image = image + epsilon * sign_data_grad
    return perturbed_image


def test(model, device, test_loader, epsilon):
    Sum = 0  # 未受攻击前可以被正确分类的样本总数
    Success = 0  # 依照要求成功完成攻击任务的样本数  (init_pred +1)%10 = final_pred
    adv_examples = []
    count = 1  # 总样本计数
    for data, target in test_loader:
        data, target = data.to(device), target.to(device)
        data.requires_grad = True
        output = model(data)
        init_pred = output.max(1, keepdim=True)[1]
        if init_pred.item() != target.item():
            print("Sample {} is a improper sample".format(count))  # 只有初始分类正确的样本才有攻击的必要
            count += 1
            continue
        Sum += 1
        loss = F.nll_loss(output, target)
        model.zero_grad()
        loss.backward()
        data_grad = data.grad.data
        perturbed_data = fgsm_attack(data, epsilon, data_grad)
        output = model(perturbed_data)
        final_pred = output.max(1, keepdim=True)[1]
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


parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
parser.add_argument('--attack-batch-size', type=int, default=1, metavar='N',
                    help='input batch size for testing (default: 1000)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
args = parser.parse_args()
test_loader = torch.utils.data.DataLoader(
    datasets.FashionMNIST('../data', train=False, download=True,
                          transform=transforms.Compose([
                              transforms.ToTensor(),
                              transforms.Normalize((0.1307,), (0.3081,))
                          ])),
    batch_size=args.attack_batch_size, shuffle=True)
use_cuda = not args.no_cuda and torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
acc, ex = test(model, device, test_loader, epsilons)
print("Saving List in WhiteAttackOnPost_data.npy")
examples = np.array(ex)
np.save('WhiteAttackOnPost_data.npy', examples)
# Epsilon: 0.3	Success_rate = 410 / 9082 = 0.0451442413565294
