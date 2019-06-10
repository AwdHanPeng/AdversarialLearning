# 在新分类器上重复黑盒攻击 且黑盒攻击的使用样本为测试集
# 新分类器为BlackNet
# 黑盒攻击时使用的迁移白盒为WhiteNet 其模型定义于WhiteAttackModel.py文件中
import torch
import torch.nn.functional as F
import argparse
import numpy as np
from WhiteAttackModel import Net as WhiteNet
from model import Net as BlackNet

WhiteNet = WhiteNet()
WhiteNet.load_state_dict(torch.load("white_model.pt"))

BlackNet = BlackNet()
BlackNet.load_state_dict(torch.load("new_fashion_mnist_cnn.pt"))
epsilons = .3


def fgsm_attack(image, epsilon, data_grad):
    sign_data_grad = data_grad.sign()
    perturbed_image = image + epsilon * sign_data_grad
    return perturbed_image


def test(device, test_loader, epsilon):
    Sum = 0  # 总样本数=9082 为newmodel可以正确分类的样本总数
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
        WhiteOutput = WhiteNet(data)
        loss = F.nll_loss(WhiteOutput, target)
        Sum += 1
        WhiteNet.zero_grad()
        loss.backward()
        data_grad = data.grad.data
        perturbed_data = fgsm_attack(data, epsilon, data_grad)
        PerturbedBlackOutput = BlackNet(perturbed_data)
        final_pred = PerturbedBlackOutput.max(1, keepdim=True)[1]
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
print("Saving List in BlackAttackOnPost_data.npy")
examples = np.array(ex)
np.save('BlackAttackOnPost_data.npy', examples)
# Epsilon: 0.3	Success_rate = 420 / 9082 = 0.046245320414005725