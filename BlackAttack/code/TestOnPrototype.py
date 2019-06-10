# 此文件用以分别测试两个模型在测试集上的表现
# WhiteNet为用于迁移的白盒模型
# BlackNet为原始待黑盒攻击的模型
import torch
import torch.nn.functional as F
from torchvision import datasets, transforms
import argparse
from WhiteAttackModel import Net as WhiteNet

WhiteNet = WhiteNet()
WhiteNet.load_state_dict(torch.load("white_model.pt"))
from model import Net as BlackNet

BlackNet = BlackNet()
BlackNet.load_state_dict(torch.load("fashion_mnist_cnn.pt"))


def test(args, model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target,
                                    reduction='sum').item()  # sum up batch loss 交叉熵本来就不用除以10 这里算的是所有样本的总损失
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                    help='input batch size for testing (default: 1000)')
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
print("WhiteNet----------------------------------")
test(args, WhiteNet, device, test_loader)
# Test set: Average loss: 0.2980, Accuracy: 8923/10000 (89%)
print("BlackNet----------------------------------")
test(args, BlackNet, device, test_loader)
# Test set: Average loss: 0.2844, Accuracy: 8989/10000 (90%)
