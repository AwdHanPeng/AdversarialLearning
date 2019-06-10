import matplotlib.pylab as plt
import numpy as np
examples = np.load('data.npy')
examples = examples.tolist()
plt.figure(figsize=(8,10))
for i in range(10):
    init_pred, pre_ex, final_pred, adv_ex = examples[i]
    plt.subplot(10, 2, i * 2 + 1)
    plt.title("init_pred->{}".format(init_pred))
    plt.imshow(pre_ex)
    plt.subplot(10, 2, i * 2 + 2)
    plt.title("final_pred->{}".format(final_pred))
    plt.imshow(adv_ex)
plt.show()
