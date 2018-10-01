import tensorflow as tf
import numpy as np
from sklearn.metrics import confusion_matrix
import MLtest

train_samples, train_labels = MLtest._train_samples, MLtest._train_labels
test_samples, test_labels = MLtest._test_samples, MLtest.test_labels

#print('Train data:', train_samples.shape, train_labels.shape)
#print('Test data: ', test_samples.shape, test_labels.shape)

image_size = MLtest.image_size
num_labels = MLtest.num_labels
num_channels = MLtest.num_channels


# Iterator: to get a batch size of data each time
# 用于 for loop，range() function
def get_chunk(samples, labels, chunk_size):
    if len(samples) != len(labels):
        raise Exception('Length of samples and labels must equal')
    stepStart = 0
    i = 0
    while stepStart < len(samples):
        stepEnd = stepStart + chunk_size
        if stepEnd < len(samples):
            yield i, samples[stepStart:stepEnd], labels[stepStart:stepEnd]
            i += 1
        stepStart = stepEnd

class Network():
    # step 1. 初始化
    # @hidden_num: 隐藏层的节点数量
    # @batch_size: 为了结束内存，分批处理数据，每一批的数据量
    # @patch_size: 滑窗的size
    def __init__(self, hidden_num, batch_size, patch_size, conv_depth, pooling_scale):

        self.batch_size = batch_size
        self.test_batch_size = 500

        # Hyper parameters: 在隐藏层调参数
        self.hidden_num = hidden_num
        self.patch_size = patch_size
        self.conv1_depth = conv_depth
        self.conv2_depth = conv_depth
        self.conv3_depth = conv_depth
        self.conv4_depth = conv_depth
        self.last_conv_depth = self.conv4_depth
        self.pooling_scale = pooling_scale
        self.pooling_stride = self.pooling_scale   # Max Pooling skill

        # Graph related
        self.graph = tf.Graph()
        self.tf_train_samples = None
        self.tf_train_labels = None
        self.tf_test_samples = None
        self.tf_test_labels = None
        self.tf_test_prediction = None

        # 统计
        self.merged = None

        # initialize
        self.define_graph()
        self.session = tf.Session(graph=self.graph)
        self.writer = tf.summary.FileWriter(' ./tfboard', self.graph)


    # step 2. 定义计算图谱
    def define_graph(self):

        with self.graph.as_default():
            # 定义图谱中的各种变量
            with tf.name_scope('input'):
                self.tf_train_samples = tf.placeholder(tf.float32, shape=(self.batch_size, image_size, image_size, num_channels), name='tf_train_samples')
                self.tf_train_labels = tf.placeholder(tf.float32, shape=(self.batch_size, num_labels), name='tf_train_labels')
                self.tf_test_samples = tf.placeholder(tf.float32, shape=(self.test_batch_size, image_size, image_size, num_channels), name='tf_test_samples')

            # 定义卷积神经网络个层的参数和变量
            with tf.name_scope('conv1'):
                # @self.conv1_depth：通过第一个filter输出有几层
                conv1_weights = tf.Variable(
                    tf.truncated_normal([self.patch_size, self.patch_size, num_channels, self.conv1_depth], stddev=0.1)
                )
                # 在这里bias 被设置为 0 了
                conv1_biases = tf.Variable(tf.zeros([self.conv1_depth]))

            with tf.name_scope('conv2'):
                conv2_weights = tf.Variable(
                    tf.truncated_normal([self.patch_size, self.patch_size, self.conv1_depth, self.conv2_depth], stddev=0.1)
                )
                conv2_biases = tf.Variable(tf.constant(0.1, shape=[self.conv2_depth]))

            with tf.name_scope('conv3'):
                conv3_weights = tf.Variable(
                    tf.truncated_normal([self.patch_size, self.patch_size, self.conv2_depth, self.conv3_depth], stddev=0.1)
                )
                conv3_biases = tf.Variable(tf.constant(0.1, shape=[self.conv3_depth]))

            with tf.name_scope('conv4'):
                conv4_weights = tf.Variable(
                    tf.truncated_normal([self.patch_size, self.patch_size, self.conv3_depth, self.conv4_depth], stddev=0.1)
                )
                conv4_biases = tf.Variable(tf.constant(0.1, shape=[self.conv4_depth]))



            # fully connected network:
            # Max polling: 在卷积层处理完后，对图像进行压缩（损失精度）
            #             如果原图：10 * 10；pooling_stride = 2; 那么现在是 5 * 5
            # hidden layer (layer 1)
            with tf.name_scope('fc1'):
                down_scale = self.pooling_stride ** 2    # 做两次pooling stride
                fc1_weights = tf.Variable(
                    # 标准差为0.1的正态分布   长 * 宽 * 一次的图片数量 = 上一层的输入量，     本层的节点数量    标准差
                    tf.truncated_normal([(image_size//down_scale) * (image_size//down_scale) * self.last_conv_depth , self.hidden_num], stddev=0.1)
                )
                fc1_biases = tf.Variable(
                    tf.constant(0.1, shape=[self.hidden_num])
                )

                tf.summary.histogram('fc1_weights', fc1_weights)
                tf.summary.histogram('fc1_biases', fc1_biases)

                # output layer (layer 2)
            with tf.name_scope('fc2'):
                fc2_weights = tf.Variable(
                    tf.truncated_normal([self.hidden_num, num_labels], stddev=0.1)
                )
                fc2_biases = tf.Variable(tf.constant(0.1, shape=[num_labels]))

                tf.summary.histogram('fc2_weights', fc2_weights)
                tf.summary.histogram('fc2_biases', fc2_biases)

                # 定义上面graph里面的每一层到底怎么运算的
                def model(samples):
                    # @data: original inputs
                    # @return: logits

                    # conv1 和 conv2 为一组； conv3 和 conv4 为一组
                    with tf.name_scope('conv1_model'):
                        conv1 = tf.nn.conv2d(samples, filter=conv1_weights, strides=[1, 1, 1, 1], padding='SAME')
                        hidden = tf.nn.relu(conv1 + conv1_biases)

                    with tf.name_scope('conv2_model'):
                        # 这里strides是conv2里面的slide_window
                        conv2 = tf.nn.conv2d(hidden, filter=conv2_weights, strides=[1, 1, 1, 1], padding='SAME')
                        hidden = tf.nn.relu(conv2 + conv2_biases)

                        # 这里的strides是max pooling的slide_window; batch = 1(每次处理一张)；channels = 1
                        hidden = tf.nn.max_pool(
                            hidden,
                            ksize=[1, self.pooling_scale, self.pooling_scale, 1],
                            strides=[1, self.pooling_stride, self.pooling_stride, 1],
                            padding='SAME')

                    with tf.name_scope('conv3_model'):
                        conv3 = tf.nn.conv2d(hidden, filter=conv3_weights, strides=[1, 1, 1, 1], padding='SAME')
                        hidden = tf.nn.relu(conv3+conv3_biases)

                    with tf.name_scope('conv4_model'):
                        conv4 = tf.nn.conv2d(hidden, filter=conv4_weights, strides=[1, 1, 1, 1], padding='SAME')
                        hidden = tf.nn.relu(conv4+conv4_biases)
                        hidden = tf.nn.max_pool(
                            hidden,
                            ksize=[1, self.pooling_scale, self.pooling_scale, 1],
                            strides=[1, self.pooling_stride, self.pooling_stride, 1],
                            padding='SAME')

                    # input layer -> hidden layer (layer 1)
                    # 把数据flat成二维的
                    shape = hidden.get_shape().as_list()
                    # print(samples.get_shape(), shape)
                    reshape = tf.reshape(hidden, [shape[0], shape[1] * shape[2] * shape[3]])

                    with tf.name_scope('fc1_model'):
                        # a1 * w + b and active function is ReLu
                        fc1 = tf.matmul(reshape, fc1_weights) + fc1_biases
                        hidden = tf.nn.relu(fc1)

                    # hidden layer -> output layer (layer 2)
                    with tf.name_scope('fc2_model'):
                        return tf.matmul(hidden, fc2_weights) + fc2_biases

                # Training computation
                logits = model(self.tf_train_samples)
                with tf.name_scope('loss'):
                    self.loss = tf.reduce_mean(
                        tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=self.tf_train_labels)
                    )
                    tf.summary.scalar('loss', self.loss)

                # Optimizer
                with tf.name_scope('optimizer'):
                    self.optimizer = tf.train.GradientDescentOptimizer(0.0001).minimize(self.loss)

                # Predictions for the training, validation, and test data
                with tf.name_scope('predictions'):
                    self.train_prediction = tf.nn.softmax(logits, name='train_prediction')
                    self.test_prediction = tf.nn.softmax(model(self.tf_test_samples), name='test_prediction')

                self.merged = tf.summary.merge_all()

    # 简单训练 + 测试
    def run(self):
        # must use session to run the network

        # private function
        def print_confusion_matrix(confusionMatrix):
            print('Confusion     Matrix:')
            for i, line in enumerate(confusionMatrix):
                print(line, line[i]/np.sum(line))
            a = 0
            for i, column in enumerate(np.transpose(confusionMatrix, (1, 0))):
                a += (column[i]/np.sum(column)) * (np.sum(column)/26000)
                print(column[i]/np.sum(column),)
            print('\n', np.sum(confusionMatrix), a)


        with self.session as session:
            tf.global_variables_initializer().run()

            # Start training
            print('Training Start')
            # batch = 1000
            for i, sample, label in get_chunk(train_samples, train_labels, chunk_size=self.batch_size):
                # '_' means we don't need to store the optimizer's result
                _, l, predictions, summary = session.run(
                    [self.optimizer, self.loss, self.train_prediction, self.merged],
                    feed_dict={self.tf_train_samples: sample, self.tf_train_labels: label}
                )
                self.writer.add_summary(summary, i)
                accuracy, _ = self.accuracy(predictions, label)
                if i % 50 == 0:
                    print('Minibatch loss at step %d: %f' % (i, l))
                    print('Minibatch accuracy: %.1f%%' % accuracy)


            # 测试
            accuracies = []
            confusionMatrices = []
            for i, sample, label in get_chunk(test_samples, test_labels, chunk_size=self.test_batch_size):
                result = self.test_prediction.eval(feed_dict={self.tf_test_samples: sample})
                accuracy, cm = self.accuracy(result, label, need_confusion_matrix=True)
                accuracies.append(accuracy)
                confusionMatrices.append(cm)
                print('Test Accuracy: %. 1f%%' % accuracy)
            print('Average accuracy:', np.average(accuracies))
            print('Standard deviation: ', np.std(accuracies))
            print_confusion_matrix(np.add.reduce(confusionMatrices))


    def accuracy(self, predictions, labels, need_confusion_matrix=False):
        # 计算预测的正确率和召回率
        # @return：accuacy and confusionMatrix as a tuple
        _predictions = np.argmax(predictions, 1)
        _labels = np.argmax(labels, 1)
        cm = confusion_matrix(_labels, _predictions) if need_confusion_matrix else None
        # if a = [3, 2, 1], b = [1, 2, 3]
        # a = np.array(a), b = np.array(b); a == b -> array([false, true, false])
        # np.sum(a == b) -> 1 , since there is only one true.

        #            100  * 正确的个数 / 所有图片的个数
        accuracy = (100.0 * np.sum(_predictions == labels) / predictions.shape[0])
        return accuracy, cm


if __name__ == '__main__':
    net = Network(hidden_num=16, batch_size=32, patch_size=3, conv_depth=16, pooling_scale=2)
    net.run()