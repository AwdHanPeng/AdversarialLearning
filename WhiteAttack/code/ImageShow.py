import matplotlib.pylab as plt
import numpy as np

# 此文件依据存储的对抗样本 生成图像
# 对抗样本存储于data.npy中，包含45组对抗成功的样本
# 每一组中包含（原始预测值，原始像素级数据，对抗后预测值，对抗后像素级数据）
examples = np.load('data.npy')
examples = examples.tolist()
plt.figure(figsize=(8, 10))
for i in range(10):
    init_pred, pre_ex, final_pred, adv_ex = examples[i]
    plt.subplot(10, 2, i * 2 + 1)
    plt.title("init_pred->{}".format(init_pred))
    plt.imshow(pre_ex)
    plt.subplot(10, 2, i * 2 + 2)
    plt.title("final_pred->{}".format(final_pred))
    plt.imshow(adv_ex)
plt.show()
