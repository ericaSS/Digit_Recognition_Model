import numpy as np
from scipy.io import loadmat as load
import matplotlib.pyplot as plt


# Machine Learning Project (Graph Recognize)
# 1. 下载数据
# 2. 探索数据
# 3. 预处理数据
# 4. 构建一个基本神经网络
# 5. 卷积
# 6. 实验
# 7. 微调与结果

traindata = load('../tensorflow/data/train_32x32.mat')
testdata = load('../tensorflow/data/test_32x32.mat')
#extradata = load("../tensorflow/data/extra_32x32.mat")

train_samples = traindata['X']
train_labels = traindata['y']
test_samples = testdata['X']
test_labels = testdata['y']

num_labels = 10 # there are 10 digits from 0 to 9image_size = 32
num_channels = 1
image_size = 32

# print('Train data samples shape:', traindata['X'].shape)
# print('Train data labels shape:', traindata['y'].shape)
#
# print('Test data sample shape:', testdata['X'].shape)
# print('Test data labels shape:', testdata['y'].shape)

# print('Extra data sample shape:', extradata['X'].shape)
# print('Extra data labels shape:', extradata['X'].shape)


# step 1: to reformat the data
# sample reformat：.mat (图片高，图片宽，通道数，图片个数）-> (图片个数，图片高，图片宽，通道数）
# label reformat: change to one-hot encoding ex.(0 0 1 0 0 0 0 0 0 0) -> 2
def reformat(samples, labels):
    new = np.transpose(samples, (3, 0, 1, 2)) # samples reformat
    labels = np.array(list(x[0] for x in labels))
    # bug because the np.array does not take an iterator, need to convert to a list
    one_hot_labels = []
    for num in labels:
        one_hot = [0.0] * 10
        if num == 10:
            one_hot[0] = 1.0
        else:
            one_hot[num] = 1.0
        one_hot_labels.append(one_hot)
    labels = np.array(one_hot_labels).astype(np.float32)
    return new, labels

# step 2: to normalize the samples for save the memory
# 灰度化(grayscaling images)： 从三色通道 -> 单色通道    省内存 + 加快训练速度
# （Red + Green + Blue) / 3
# 正则化：将图片从 0 ～ 255 线性映射到 -1.0 ~ +1.0
def normalize(samples):
    # (图片个数， 图片高， 图片宽， 通道数) 沿着通道数相加
    a = np.add.reduce(samples, keepdims=True, axis=3)
    a = a/3.0
    return a/128.0 - 1.0


# step 3: to check the each label's distribution
# 查看每个label的分布，并且画一个统计图
# 0 - 9
def distribution(labels, name):
    count = {}
    for label in labels:
        key = 0 if label[0] == 10 else label[0]
        if key in count:
            count[key] += 1
        else:
            count[key] = 1
    # 画一个统计图
    x = []
    y = []
    # k = key, v = value
    for k, v in count.items():
       # print(k, v)
        x.append(k)
        y.append(v)

    # 设计图的样式
    y_axis = np.arange(len(x))
    plt.bar(y_axis, y, align='center', alpha=0.5)
    plt.xticks(y_axis, x) # （y_axis, x）x轴上数值是乱序的，改为（x, x) x轴上从1-9, 但是和预想中的分配不一样
    plt.ylabel('count')
    plt.title(name + 'Distribution')
    plt.show()


# 来查看图片，并且给出对应的label
def inspect(samples, labels, i):
    if samples.shape[3] == 1:
        shape = samples.shape
        samples = samples.reshape(shape[0],shape[1],shape[2])
    print(labels[i])
    plt.imshow(samples[i])
    plt.show()

tr_samples, _train_labels = reformat(train_samples, train_labels)
te_samples, _test_labels = reformat(test_samples, test_labels)

_train_samples = normalize(tr_samples)
_test_samples = normalize(te_samples)

if __name__ == '__main__':
    pass
    # print(tr_samples.shape)
    # inspect(tr_samples, tr_labels, 1)
    # inspect(_train_samples, tr_labels, 1)
    #distribution(train_labels, 'Train labels ')
    #distribution(test_labels, 'Test labels ')

