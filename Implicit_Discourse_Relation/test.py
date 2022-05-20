import numpy as np

trainFiled1 = np.loadtxt('test_pred-2.txt', dtype=int, delimiter="\t")
# trainFiled2 = np.loadtxt('2.txt', dtype=int, delimiter="\t")
trainFiled3 = np.loadtxt('IDRR实验结果_信卓1901_U201913504_王浩芃6000.txt', dtype=int, delimiter="\t")
resnum = 0
for i in range(2000):
    if trainFiled1[i] == trainFiled3[i]:
            resnum += 1
# np.savetxt('vote.txt', res, fmt='%d', delimiter='\n')
print(resnum)