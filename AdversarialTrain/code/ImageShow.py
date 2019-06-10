import matplotlib.pylab as plt
import numpy as np

# 在新分类器上白盒攻击成功的样本中挑选十组
white_examples = np.load('WhiteAttackOnPost_data.npy')
white_examples = white_examples.tolist()
plt.figure(figsize=(8, 10))
for i in range(10):
    init_pred, pre_ex, final_pred, adv_ex = white_examples[i]
    plt.subplot(10, 2, i * 2 + 1)
    plt.title("White_init_pred->{}".format(init_pred))
    plt.imshow(pre_ex)
    plt.subplot(10, 2, i * 2 + 2)
    plt.title("White_final_pred->{}".format(final_pred))
    plt.imshow(adv_ex)
plt.show()
# 在新分类器上黑盒攻击成功的样本中挑选十组
black_examples = np.load('BlackAttackOnPost_data.npy')
black_examples = black_examples.tolist()
plt.figure(figsize=(8, 10))
for i in range(10):
    init_pred, pre_ex, final_pred, adv_ex = black_examples[i]
    plt.subplot(10, 2, i * 2 + 1)
    plt.title("black_init_pred->{}".format(init_pred))
    plt.imshow(pre_ex)
    plt.subplot(10, 2, i * 2 + 2)
    plt.title("black_final_pred->{}".format(final_pred))
    plt.imshow(adv_ex)
plt.show()

