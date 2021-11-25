import sys
from xlrd import open_workbook
from xlutils.copy import copy
from xlwt import Workbook
from sklearn import preprocessing
import os
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
import pandas as pd
import scipy.io as sio
import tensorflow as tf
import numpy as np
import time
import math

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

def file_name(file_dir="D:/CB/deapProject/result/modelSubIndependent_result/total_result/"):
    for root, dirs, files in os.walk(file_dir):
        print(files)
    return files

input_height = 1
input_width = 32
input_channel_num = 1
lambda_loss=0.01
epsilon = 1e-14

time_step = 1
window_size = 1
# convolution full connected parameter
fc_size = 1024
dropout_prob = 0.5
np.random.seed(3)

calibration = 'N'
norm_type = '2D'
regularization_method = 'dropout'
enable_penalty = True

args = sys.argv[:]
# with_or_not = args[1]
with_or_not= 'with'
arousal_or_valence= 'dominance'
inputs = list(map(int, args[3:]))
bands=[0,1,2,3]
print(bands)

dataset_dir = "D:/CB/deapProject/DE_feature/DE_feature_allSub/"
###load training set
print("loading ", dataset_dir + "DE_allSub.mat")
data_file = sio.loadmat(dataset_dir + "DE_allSub.mat")

data = data_file["data"]
datasets_theta = data[:, 0:32].reshape(-1,input_height, input_width, input_channel_num)
datasets_alpha = data[:,32:64].reshape(-1,input_height, input_width, input_channel_num)
datasets_beta = data[:,64:96].reshape(-1,input_height, input_width, input_channel_num)
datasets_gamma = data[:,96:128].reshape(-1,input_height, input_width, input_channel_num)
label_key = arousal_or_valence + "_labels"
labels = data_file[label_key]

# 2018-5-16 modified
label_index = [i for i in range(0, labels.shape[1], time_step)]

labels = labels[0, [label_index]]
labels = np.squeeze(np.transpose(labels))
one_hot_labels = np.array(list(pd.get_dummies(np.transpose(labels))))

labels = np.asarray(pd.get_dummies(labels), dtype=np.int8)
# shuffle data
index = np.array(range(0, len(labels)))
np.random.shuffle(index)

datasets_theta = datasets_theta[index]
datasets_alpha = datasets_alpha[index]
datasets_beta = datasets_beta[index]
datasets_gamma = datasets_gamma[index]

labels = labels[index]

print("**********(" + time.asctime(time.localtime(time.time())) + ") Load and Split dataset End **********\n")
print(
    "**********(" + time.asctime(time.localtime(time.time())) + ") Define parameters and functions Begin: **********\n")
n_labels = 2
# training parameter
lambda_loss_amount = 0.01
training_epochs = 10
batch_size = 1200
# kernel parameter
kernel_height_1st = 3
kernel_width_1st = 3

kernel_height_2nd = 3
kernel_width_2nd = 3

kernel_height_3rd = 3
kernel_width_3rd = 3

kernel_height_4th = 1
kernel_width_4th = 1

kernel_stride = 1
conv_channel_num = 64
# pooling parameter
pooling_height = 2
pooling_width = 2
pooling_stride = 2
# algorithn parameter
learning_rate = 1e-4

# 参数概要
def variable_summary(var):
    with tf.name_scope("summaries"):
        mean = tf.reduce_mean(var)
        tf.summary.scalar("mean", mean)  # 平均值
    with tf.name_scope("stddev"):
        stddev = tf.sqrt(tf.reduce_mean(var - mean))
    tf.summary.scalar("stddev", stddev)  # 标准差
    tf.summary.scalar("max", tf.reduce_max(var))  # 最大值
    tf.summary.scalar("min", tf.reduce_min(var))  # 最小值
    tf.summary.histogram("histogram", var)  # 直方图

def hw_flatten(x):
    shape=x.get_shape().as_list()
    x_flatten=tf.reshape(x,[-1,shape[1]*shape[2],shape[3]])
    return x_flatten

def attention(x, ch, scope='attention', reuse=tf.AUTO_REUSE):
    with tf.variable_scope(scope, reuse=reuse):
        f = apply_conv2d(x, 1, 1, ch // 8, 1, name='f_conv')  # [bs, h, w, c'] (?,1,32,32)
        g = apply_conv2d(x, 1, 1, ch // 8, 1, name='g_conv')  # [bs, h, w, c']
        h = apply_conv2d(x, 1, 1, ch , 1, name='h_conv')  # [bs, h, w, c] (?,1,32,256)
        # N = h * w
        s = tf.matmul(hw_flatten(g), hw_flatten(f), transpose_b=True)  # # [bs, N, N]
        beta = tf.nn.softmax(s, axis=-1)  # attention map

        o = tf.matmul(beta, hw_flatten(h))  # [bs, N, C]
        gamma = tf.get_variable("gamma", [1], initializer = tf.constant_initializer(0.0))

        o = tf.reshape(o, shape=[-1, input_height, input_width, ch])  # [bs, h, w, C]
        o = apply_conv2d(o, 1, 1, ch, 1, name='attn_conv')
        x = gamma * o + x

    return x

def conv2d(x, W, kernel_stride):
    # API: must strides[0]=strides[4]=1
    return tf.nn.conv2d(x, W, strides=[1, kernel_stride, kernel_stride, 1], padding='SAME')

def apply_conv2d(x, filter_height, filter_width, out_channels, kernel_stride,name='conv'):
    with tf.variable_scope(name):
        weight=tf.get_variable('weight',shape=[filter_height, filter_width, x.get_shape()[-1], out_channels],
                               initializer=tf.random_normal_initializer(stddev=0.1))
        bias = tf.get_variable('bias',[out_channels],initializer=tf.constant_initializer(0.1))  # each feature map shares the same weight and bias
        conv=tf.add(conv2d(x,weight,kernel_stride),bias,name='conv_add_b')
        return tf.nn.selu(conv)

def apply_fully_connect(x, x_size, fc_size, name='fully_connect'):
    with tf.variable_scope(name):
        fc_weight = tf.get_variable('fc_weight',[x_size, fc_size],
                                    initializer=tf.random_normal_initializer(stddev=0.1))
        mul = tf.matmul(x, fc_weight, name='fc_mul')
        fc_bias = tf.get_variable('fc_bias',[fc_size],initializer=tf.constant_initializer(0.1))
        fc_mul = tf.add(mul, fc_bias, name="mul_add_b")
        return tf.nn.selu(fc_mul)

def apply_readout(x, x_size, readout_size,name='readout'):
    with tf.variable_scope(name):
        readout_weight = tf.get_variable('readout_weight',[x_size, readout_size],
                                         initializer=tf.random_normal_initializer(stddev=0.1))
        variable_summary(readout_weight)
        mul = tf.matmul(x, readout_weight, name="read_mul")
        readout_bias = tf.get_variable('readout_bias',[readout_size],initializer=tf.constant_initializer(0.1))
        readout_mul = tf.add(mul, readout_bias, name="readout_add_b")
        return readout_mul

print("\n**********(" + time.asctime(time.localtime(time.time())) + ") Define parameters and functions End **********")
print("\n**********(" + time.asctime(time.localtime(time.time())) + ") Define NN structure Begin: **********")

# input placeholder
theta_in = tf.placeholder(tf.float32, shape=[None, input_height, input_width, input_channel_num], name='theta_in')
alpha_in = tf.placeholder(tf.float32, shape=[None, input_height, input_width, input_channel_num], name='alpha_in')
beta_in = tf.placeholder(tf.float32, shape=[None, input_height, input_width, input_channel_num], name='beta_in')
gamma_in = tf.placeholder(tf.float32, shape=[None, input_height, input_width, input_channel_num], name='gamma_in')
Y = tf.placeholder(tf.float32, shape=[None, n_labels], name='Y')

keep_prob = tf.placeholder(tf.float32, name='keep_prob')
phase_train = tf.placeholder(tf.bool, name='phase_train')
###########################################################################################
# add cnn parallel to network
###########################################################################################
with tf.name_scope("conv1_theta"):
    conv1_theta = apply_conv2d(theta_in, kernel_height_1st, kernel_width_1st, conv_channel_num, kernel_stride,name='conv1_theta')
with tf.name_scope("conv2_theta"):
    conv2_theta = apply_conv2d(conv1_theta, kernel_height_2nd, kernel_width_2nd, conv_channel_num * 2,kernel_stride,name='conv2_theta')
with tf.name_scope("conv3_theta"):
    conv3_theta = apply_conv2d(conv2_theta, kernel_height_3rd, kernel_width_3rd, conv_channel_num * 4,kernel_stride, name='conv3_theta')# (?,1,32,256)

with tf.name_scope("conv1_alpha"):
    conv1_alpha = apply_conv2d(alpha_in, kernel_height_1st, kernel_width_1st, conv_channel_num, kernel_stride,name='conv1_alpha')
with tf.name_scope("conv2_alpha"):
    conv2_alpha = apply_conv2d(conv1_alpha, kernel_height_2nd, kernel_width_2nd, conv_channel_num * 2,kernel_stride,name='conv2_alpha')
with tf.name_scope("conv3_alpha"):
    conv3_alpha = apply_conv2d(conv2_alpha, kernel_height_3rd, kernel_width_3rd, conv_channel_num * 4,kernel_stride, name='conv3_alpha')# (?,1,32,256)

with tf.name_scope("conv1_beta"):
    conv1_beta = apply_conv2d(beta_in, kernel_height_1st, kernel_width_1st, conv_channel_num, kernel_stride,name='conv1_beta')
with tf.name_scope("conv2_beta"):
    conv2_beta = apply_conv2d(conv1_beta, kernel_height_2nd, kernel_width_2nd, conv_channel_num * 2,kernel_stride,name='conv2_beta')
with tf.name_scope("conv3_beta"):
    conv3_beta = apply_conv2d(conv2_beta, kernel_height_3rd, kernel_width_3rd, conv_channel_num * 4,kernel_stride, name='conv3_beta')# (?,1,32,256)

with tf.name_scope("conv1_gamma"):
    conv1_gamma = apply_conv2d(gamma_in, kernel_height_1st, kernel_width_1st, conv_channel_num, kernel_stride, name='conv1_gamma')
with tf.name_scope("conv2_gamma"):
    conv2_gamma = apply_conv2d(conv1_gamma, kernel_height_2nd, kernel_width_2nd, conv_channel_num * 2,kernel_stride, name='conv2_gamma')
with tf.name_scope("conv3_gamma"):
    conv3_gamma = apply_conv2d(conv2_gamma, kernel_height_3rd, kernel_width_3rd, conv_channel_num * 4,kernel_stride, name='conv3_gamma')# (?,1,32,256)

theta_data=attention(conv3_theta,conv_channel_num * 4)   #(?,1,32,256)
alpha_data=attention(conv3_alpha,conv_channel_num * 4)   #(?,1,32,256)
beta_data=attention(conv3_beta,conv_channel_num * 4)   #(?,1,32,256)
gamma_data=attention(conv3_gamma,conv_channel_num * 4)  #(?,1,32,256)

theta_shape = theta_data.get_shape().as_list()
alpha_shape = alpha_data.get_shape().as_list()
beta_shape = beta_data.get_shape().as_list()
gamma_shape = gamma_data.get_shape().as_list()

theta_flat = tf.reshape(theta_data,[-1, theta_shape[1] * theta_shape[2] * theta_shape[3]])
alpha_flat=tf.reshape(alpha_data,[-1, theta_shape[1] * theta_shape[2] * theta_shape[3]])
beta_flat=tf.reshape(beta_data,[-1, theta_shape[1] * theta_shape[2] * theta_shape[3]])
gamma_flat=tf.reshape(gamma_data,[-1, theta_shape[1] * theta_shape[2] * theta_shape[3]])
total_flat=tf.concat([theta_flat,alpha_flat,beta_flat,gamma_flat],axis=1)

total_fc=apply_fully_connect(total_flat,total_flat.shape[1],fc_size,name='total_fc')
total_fc_drop=tf.nn.dropout(total_fc,keep_prob)

y_ = apply_readout(total_fc_drop, total_fc_drop.shape[1], n_labels, name='readout')
y_pred = tf.argmax(tf.nn.softmax(y_), 1, name="y_pred")
y_posi = tf.nn.softmax(y_, name="y_posi")

# l2 regularization
l2 = lambda_loss_amount * sum(
    tf.nn.l2_loss(tf_var) for tf_var in tf.trainable_variables()
)

if enable_penalty:
    # cross entropy cost function
    # cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y_, labels=Y)+ l2 , name='loss')
    cost = tf.reduce_mean(-tf.reduce_sum(Y * tf.log(tf.clip_by_value(y_posi , 1e-8,1.0)), reduction_indices=[1]))
    # cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y_, labels=Y) + l2, name='loss')
else:
    # cross entropy cost function
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y_, labels=Y), name='loss')

tf.summary.scalar('loss',cost)
tf.summary.scalar("learningRate",learning_rate)

optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)
# optimizer=tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)
# get correctly predicted object and accuracy
correct_prediction = tf.equal(tf.argmax(tf.nn.softmax(y_), 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name='accuracy')

tf.summary.scalar('accuracy',accuracy)
for var in tf.trainable_variables():
    tf.summary.histogram(var.name, var)

# Merge all summaries into a single op
merged_summary_op=tf.summary.merge_all()
saver=tf.train.Saver()

print("\n**********(" + time.asctime(time.localtime(time.time())) + ") Define NN structure End **********")
print("\n**********(" + time.asctime(time.localtime(time.time())) + ") Train and Test NN Begin: **********")
# run
config = tf.ConfigProto()
config.gpu_options.allow_growth = True

fold = 32
sum_acc = 0
sum_recall = 0
sum_precision = 0
sum_f1 = 0
sum_auc = 0
for curr_fold in range(0, fold):
    print("folder: ", curr_fold)
    fold_size = datasets_theta.shape[0] // fold
    indexes_list = [i for i in range(len(datasets_theta))]
    indexes = np.array(indexes_list)
    split_list = [i for i in range(curr_fold * fold_size, (curr_fold + 1) * fold_size)]
    split = np.array(split_list)
    theta_test_x = datasets_theta[split]
    alpha_test_x = datasets_alpha[split]
    beta_test_x = datasets_beta[split]
    gamma_test_x = datasets_gamma[split]
    test_y = labels[split]

    split = np.array(list(set(indexes_list) ^ set(split_list)))
    theta_train_x = datasets_theta[split]
    alpha_train_x = datasets_alpha[split]
    beta_train_x = datasets_beta[split]
    gamma_train_x = datasets_gamma[split]
    train_y = labels[split]
    train_sample = train_y.shape[0]

    # shuffle data
    index = np.array(range(0, len(train_y)))
    np.random.shuffle(index)

    theta_train_x = theta_train_x[index]
    alpha_train_x = alpha_train_x[index]
    beta_train_x = beta_train_x[index]
    gamma_train_x = gamma_train_x[index]
    train_y = train_y[index]

    print("training examples:", train_sample)
    test_sample = test_y.shape[0]
    print("test examples    :", test_sample)
    # set train batch number per epoch
    batch_num_per_epoch = math.floor(theta_train_x.shape[0] / batch_size) + 1
    # set test batch number per epoch
    accuracy_batch_size = batch_size
    train_accuracy_batch_num = batch_num_per_epoch
    test_accuracy_batch_num = math.floor(theta_test_x.shape[0] / batch_size) + 1

    with tf.Session(config=config) as session:

        # writer = tf.summary.FileWriter("D:/CB/My_experiment_Deap/final_cnnModel_log", session.graph)
        session.run(tf.global_variables_initializer())

        train_accuracy_save = np.zeros(shape=[0], dtype=float)
        test_accuracy_save = np.zeros(shape=[0], dtype=float)
        test_loss_save = np.zeros(shape=[0], dtype=float)
        train_loss_save = np.zeros(shape=[0], dtype=float)
        for epoch in range(training_epochs):
            print("learning rate: ", learning_rate)
            cost_history = np.zeros(shape=[0], dtype=float)
            for b in range(batch_num_per_epoch):
                start = b * batch_size
                if (b + 1) * batch_size > train_y.shape[0]:
                    offset = train_y.shape[0] % batch_size
                else:
                    offset = batch_size
                # offset = (b * batch_size) % (train_y.shape[0] - batch_size)
                # print("start->end:",start,"->",start+offset)
                theta_batch = theta_train_x[start:(start + offset), :, :, :]
                theta_batch = theta_batch.reshape(len(theta_batch) * window_size, input_height, input_width,
                                                  input_channel_num)
                alpha_batch = alpha_train_x[start:(start + offset), :, :, :]
                alpha_batch = alpha_batch.reshape(len(alpha_batch) * window_size, input_height, input_width,
                                                  input_channel_num)
                beta_batch = beta_train_x[start:(start + offset), :, :, :]
                beta_batch = beta_batch.reshape(len(beta_batch) * window_size, input_height, input_width,
                                                  input_channel_num)
                gamma_batch = gamma_train_x[start:(start + offset), :, :, :]
                gamma_batch = gamma_batch.reshape(len(gamma_batch) * window_size, input_height, input_width,
                                                  input_channel_num)
                # print("cnn_batch shape:",cnn_batch.shape)
                batch_y = train_y[start:(offset + start), :]
                _, c = session.run([optimizer, cost ],
                                   feed_dict={theta_in: theta_batch,alpha_in: alpha_batch, beta_in: beta_batch, gamma_in:gamma_batch, Y: batch_y, keep_prob: 1 - dropout_prob,
                                              phase_train: True})
                summary, _ = session.run([merged_summary_op, optimizer], feed_dict={theta_in: theta_batch,alpha_in: alpha_batch, beta_in: beta_batch, gamma_in:gamma_batch, Y: batch_y, keep_prob: 1 - dropout_prob,
                                              phase_train: True})
                cost_history = np.append(cost_history, c)
            # writer.add_summary(summary, epoch)
            if (epoch % 1 == 0):
                train_accuracy = np.zeros(shape=[0], dtype=float)
                test_accuracy = np.zeros(shape=[0], dtype=float)
                test_loss = np.zeros(shape=[0], dtype=float)
                train_loss = np.zeros(shape=[0], dtype=float)

                for i in range(train_accuracy_batch_num):
                    start = i * batch_size
                    if (i + 1) * batch_size > train_y.shape[0]:
                        offset = train_y.shape[0] % batch_size
                    else:
                        offset = batch_size
                    # offset = (i * accuracy_batch_size) % (train_y.shape[0] - accuracy_batch_size)
                    train_theta_batch = theta_train_x[start:(start + offset), :, :, :]
                    train_theta_batch = train_theta_batch.reshape(len(train_theta_batch) * window_size, input_height,
                                                              input_width, input_channel_num)
                    train_alpha_batch = alpha_train_x[start:(start + offset), :, :, :]
                    train_alpha_batch = train_alpha_batch.reshape(len(train_alpha_batch) * window_size, input_height,
                                                                  input_width, input_channel_num)
                    train_beta_batch = beta_train_x[start:(start + offset), :, :, :]
                    train_beta_batch = train_beta_batch.reshape(len(train_beta_batch) * window_size, input_height,
                                                                  input_width, input_channel_num)
                    train_gamma_batch = gamma_train_x[start:(start + offset), :, :, :]
                    train_gamma_batch = train_gamma_batch.reshape(len(train_gamma_batch) * window_size, input_height,
                                                                  input_width, input_channel_num)
                    train_batch_y = train_y[start:(start + offset), :]

                    train_a, train_c = session.run([accuracy, cost],
                                                   feed_dict={theta_in: train_theta_batch, alpha_in:train_alpha_batch,
                                                              beta_in: train_beta_batch, gamma_in: train_gamma_batch,
                                                              Y: train_batch_y, keep_prob: 1.0,
                                                              phase_train: False})

                    # saver.save(session, "model/my_%s_%s_model"% (sub,arousal_or_valence), global_step=10)
                    # print("save the model")

                    train_loss = np.append(train_loss, train_c)
                    train_accuracy = np.append(train_accuracy, train_a)
                print("(" + time.asctime(time.localtime(time.time())) + ") Epoch: ", epoch + 1, " Training Cost: ",
                      np.mean(train_loss), "Training Accuracy: ", np.mean(train_accuracy))
                train_accuracy_save = np.append(train_accuracy_save, np.mean(train_accuracy))
                train_loss_save = np.append(train_loss_save, np.mean(train_loss))

                if (np.mean(train_accuracy) < 0.7):
                    learning_rate = 1e-4
                elif (0.7 < np.mean(train_accuracy) < 0.85):
                    learning_rate = 5e-5
                elif (0.85 < np.mean(train_accuracy)):
                    learning_rate = 1e-6

                for j in range(test_accuracy_batch_num):
                    start = j * batch_size
                    if (j + 1) * batch_size > test_y.shape[0]:
                        offset = test_y.shape[0] % batch_size
                    else:
                        offset = batch_size
                    # offset = (j * accuracy_batch_size) % (test_y.shape[0] - accuracy_batch_size)
                    test_theta_batch = theta_test_x[start:(offset + start), :, :, :]
                    test_theta_batch = test_theta_batch.reshape(len(test_theta_batch) * window_size, input_height,
                                                            input_width, input_channel_num)
                    test_alpha_batch = alpha_test_x[start:(offset + start), :, :, :]
                    test_alpha_batch = test_alpha_batch.reshape(len(test_alpha_batch) * window_size, input_height,
                                                                input_width, input_channel_num)
                    test_beta_batch = beta_test_x[start:(offset + start), :, :, :]
                    test_beta_batch = test_beta_batch.reshape(len(test_beta_batch) * window_size, input_height,
                                                                input_width, input_channel_num)
                    test_gamma_batch = gamma_test_x[start:(offset + start), :, :, :]
                    test_gamma_batch = test_gamma_batch.reshape(len(test_gamma_batch) * window_size, input_height,
                                                                input_width, input_channel_num)
                    test_batch_y = test_y[start:(offset + start), :]

                    test_a, test_c = session.run([accuracy, cost],
                                                 feed_dict={theta_in: test_theta_batch, alpha_in: test_alpha_batch,
                                                            beta_in: test_beta_batch, gamma_in: test_gamma_batch,
                                                            Y: test_batch_y, keep_prob: 1.0,
                                                            phase_train: False})

                    test_accuracy = np.append(test_accuracy, test_a)
                    test_loss = np.append(test_loss, test_c)

                print("(" + time.asctime(time.localtime(time.time())) + ") Epoch: ", epoch + 1, " Test Cost: ",
                      np.mean(test_loss), "Test Accuracy: ", np.mean(test_accuracy), "\n")
                test_accuracy_save = np.append(test_accuracy_save, np.mean(test_accuracy))
                test_loss_save = np.append(test_loss_save, np.mean(test_loss))

            # reshuffle
            index = np.array(range(0, len(train_y)))
            np.random.shuffle(index)
            theta_train_x = theta_train_x[index]
            alpha_train_x = alpha_train_x[index]
            beta_train_x = beta_train_x[index]
            gamma_train_x = gamma_train_x[index]
            train_y = train_y[index]

        test_accuracy = np.zeros(shape=[0], dtype=float)
        test_loss = np.zeros(shape=[0], dtype=float)
        test_pred = np.zeros(shape=[0], dtype=float)
        test_true = np.zeros(shape=[0, 2], dtype=float)
        test_posi = np.zeros(shape=[0, 2], dtype=float)
        for k in range(test_accuracy_batch_num):
            start = k * batch_size
            if (k + 1) * batch_size > test_y.shape[0]:
                offset = test_y.shape[0] % batch_size
            else:
                offset = batch_size
            # offset = (k * accuracy_batch_size) % (test_y.shape[0] - accuracy_batch_size)
            test_theta_batch = theta_test_x[start:(offset + start), :, :, :]
            test_theta_batch = test_theta_batch.reshape(len(test_theta_batch) * window_size, input_height, input_width,
                                                    input_channel_num)
            test_alpha_batch = alpha_test_x[start:(offset + start), :, :, :]
            test_alpha_batch = test_alpha_batch.reshape(len(test_alpha_batch) * window_size, input_height, input_width,
                                                        input_channel_num)
            test_beta_batch = beta_test_x[start:(offset + start), :, :, :]
            test_beta_batch = test_beta_batch.reshape(len(test_beta_batch) * window_size, input_height, input_width,
                                                        input_channel_num)
            test_gamma_batch = gamma_test_x[start:(offset + start), :, :, :]
            test_gamma_batch = test_gamma_batch.reshape(len(test_gamma_batch) * window_size, input_height, input_width,
                                                        input_channel_num)
            test_batch_y = test_y[start:(offset + start), :]

            test_a, test_c, test_p, test_r = session.run([accuracy, cost, y_pred, y_posi ],
                                                         feed_dict={theta_in: test_theta_batch, alpha_in: test_alpha_batch,
                                                                    beta_in: test_beta_batch,gamma_in: test_gamma_batch,
                                                                    Y: test_batch_y,
                                                                    keep_prob: 1.0, phase_train: False})
            test_t = test_batch_y

            test_accuracy = np.append(test_accuracy, test_a)
            test_loss = np.append(test_loss, test_c)
            test_pred = np.append(test_pred, test_p)
            test_true = np.vstack([test_true, test_t])
            test_posi = np.vstack([test_posi, test_r])

        # test_true = tf.argmax(test_true, 1)
        test_pred_1_hot = np.asarray(pd.get_dummies(test_pred), dtype=np.int8)
        test_true_list = tf.argmax(test_true, 1).eval()
        # recall
        test_recall = recall_score(test_true, test_pred_1_hot, average=None)
        # precision
        test_precision = precision_score(test_true, test_pred_1_hot, average=None)
        # f1 score
        test_f1 = f1_score(test_true, test_pred_1_hot, average=None)
        # fpr, tpr, auc
        fpr = {}
        tpr = {}
        roc_auc = []
        i = 0
        for key in one_hot_labels:
            fpr[key], tpr[key], _ = roc_curve(test_true[:, i], test_posi[:, i])
            roc_auc.append(auc(fpr[key], tpr[key]))
            i += 1
        # confusion matrix
        # confusion_matrix = confusion_matrix(test_true_list, test_pred)
        print("********************recall:", test_recall)
        print("*****************precision:", test_precision)
        print("******************f1_score:", test_f1)
        print("**********confusion_matrix:\n", confusion_matrix)

        print("(" + time.asctime(time.localtime(time.time())) + ") Final Test Cost: ", np.mean(test_loss),
              "Final Test Accuracy: ", np.mean(test_accuracy), "recall", np.mean(test_recall), "precision", np.mean(test_precision),
              "f1", np.mean(test_f1), "auc", np.mean(roc_auc))

    sum_acc += np.mean(test_accuracy)
    sum_recall += np.mean(test_recall)
    sum_precision += np.mean(test_precision)
    sum_f1 += np.mean(test_f1)
    sum_auc += np.mean(roc_auc)

mean_acc = sum_acc / fold
mean_recall = sum_recall / fold
mean_precision = sum_precision / fold
mean_f1 = sum_f1 / fold
mean_auc = sum_auc / fold

files = file_name()
save_file_name = arousal_or_valence + "_" + str(bands) + ".xls"
print("acc:", mean_acc, "recall:",mean_recall, "precision:",mean_precision,"f1_score:",mean_f1,
      "save path: D:/CB/deapProject/result/modelSubIndependent_result/total_result/", save_file_name)
# 创建表结构，定义列名
if save_file_name not in files:
    book = Workbook(encoding='utf-8')
    sheet1 = book.add_sheet('result')
    sheet1.write(0, 0, "acc")
    sheet1.write(0, 1, "recall")
    sheet1.write(0, 2, "precision")
    sheet1.write(0, 3, "f1_score")
    sheet1.write(0, 4, "auc")
    # sheet1.write(0,2,"without")
    # 保存Excel book.save('path/文件名称.xls')
    book.save("D:/CB/deapProject/result/modelSubIndependent_result/total_result/" + save_file_name)

#打开定义好的表结构，向其中插入值
rexcel = open_workbook(
    "D:/CB/deapProject/result/modelSubIndependent_result/total_result/" + save_file_name)  # 用wlrd提供的方法读取一个excel文件
excel = copy(rexcel)  # 用xlutils提供的copy方法将xlrd的对象转化为xlwt的对象
table = excel.get_sheet(0)  # 用xlwt对象的方法获得要操作的sheet

table.write(1, 0,  mean_acc)
table.write(1, 1,  mean_recall)
table.write(1, 2,  mean_precision)
table.write(1, 3,  mean_f1)
table.write(1, 4, mean_auc)
excel.save("D:/CB/deapProject/result/modelSubIndependent_result/total_result/" + save_file_name)
