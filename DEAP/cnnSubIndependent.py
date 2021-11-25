import sys
from xlrd import open_workbook
from xlutils.copy import copy
from xlwt import Workbook
import os
import scipy.io as sio
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
import pandas as pd
import tensorflow as tf
import numpy as np
import time
import math

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def file_name(file_dir = "D:/CB/deapProject/result/cnn_result/subIndependent_result/"):
    for root, dirs, files in os.walk(file_dir):
        print(files)
    return files

input_height = 1
input_width = 128
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
arousal_or_valence= 'liking'
inputs = list(map(int, args[3:]))
bands=[0,1,2,3]
print(bands)

dataset_dir = "D:/CB/deapProject/DE_feature/DE_feature_allSub/"
###load training set
print("loading ", dataset_dir + "DE_allSub.mat")
data_file = sio.loadmat(dataset_dir + "DE_allSub.mat")

datasets = data_file["data"].reshape(-1,input_height, input_width, input_channel_num)
label_key = arousal_or_valence + "_labels"
labels = data_file[label_key]  #(1,25600)

# 2018-5-16 modified
label_index = [i for i in range(0, labels.shape[1], time_step)] #(25600)

labels = labels[0, [label_index]]  #(1,25600)
labels = np.squeeze(np.transpose(labels)) #(25600,)
one_hot_labels = np.array(list(pd.get_dummies(labels))) #(2,)
labels = np.asarray(pd.get_dummies(labels), dtype=np.int8) #(25600,2)
# shuffle data
index = np.array(range(0, len(labels)))
np.random.shuffle(index)

datasets = datasets[index]
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

def spectral_normed_weight(w,
                           u=None,
                           num_iters=1,  # For Power iteration method, usually num_iters = 1 will be enough
                           update_collection=None,
                           with_sigma=False  # Estimated Spectral Norm
                           ):
    w_shape = w.shape.as_list()
    w_new_shape = [np.prod(w_shape[:-1]), w_shape[-1]]   #np.prod实现指定元素的乘积
    w_reshaped = tf.reshape(w, w_new_shape, name='w_reshaped')
    if u is None:
        u = tf.get_variable("u_vec", [w_new_shape[0], 1], initializer=tf.truncated_normal_initializer(),
                            trainable=False)
    # power iteration
    u_ = u
    for _ in range(num_iters):
        # ( w_new_shape[1], w_new_shape[0] ) * ( w_new_shape[0], 1 ) -> ( w_new_shape[1], 1 )
        v_ = _l2normalize(tf.matmul(tf.transpose(w_reshaped), u_))
        # ( w_new_shape[0], w_new_shape[1] ) * ( w_new_shape[1], 1 ) -> ( w_new_shape[0], 1 )
        u_ = _l2normalize(tf.matmul(w_reshaped, v_))

    u_final = tf.identity(u_, name='u_final')  # ( w_new_shape[0], 1 )
    v_final = tf.identity(v_, name='v_final')  # ( w_new_shape[1], 1 )

    u_final = tf.stop_gradient(u_final)
    v_final = tf.stop_gradient(v_final)
    sigma = tf.matmul(tf.matmul(tf.transpose(u_final), w_reshaped), v_final, name="est_sigma")
    update_u_op = tf.assign(u, u_final)
    with tf.control_dependencies([update_u_op]):
        sigma = tf.identity(sigma)
        w_bar = tf.identity(w / sigma, 'w_bar')
    if with_sigma:
        return w_bar, sigma
    else:
        return w_bar

def spectral_norm(w,num_iters=1,u=None):
    w_shape = w.shape.as_list()
    w_new_shape = [np.prod(w_shape[:-1]).astype(int), w_shape[-1]]   #np.prod实现指定元素的乘积
    w_reshaped = tf.reshape(w, w_new_shape, name='w_reshaped')
    if u is None:
        u = tf.get_variable("u_vec", [w_new_shape[0], 1], initializer=tf.truncated_normal_initializer(),
                            trainable=False)
    # power iteration
    u_ = u
    for _ in range(num_iters):
        # ( w_new_shape[1], w_new_shape[0] ) * ( w_new_shape[0], 1 ) -> ( w_new_shape[1], 1 )
        v_ = _l2normalize(tf.matmul(tf.transpose(w_reshaped), u_))
        # ( w_new_shape[0], w_new_shape[1] ) * ( w_new_shape[1], 1 ) -> ( w_new_shape[0], 1 )
        u_ = _l2normalize(tf.matmul(w_reshaped, v_))

    u_final = tf.identity(u_, name='u_final')  # ( w_new_shape[0], 1 )
    v_final = tf.identity(v_, name='v_final')  # ( w_new_shape[1], 1 )

    u_final = tf.stop_gradient(u_final)
    v_final = tf.stop_gradient(v_final)
    sigma = tf.matmul(tf.matmul(tf.transpose(u_final), w_reshaped), v_final, name="est_sigma")
    return tf.square(sigma)

def _l2normalize(v, eps=1e-12):
    with tf.name_scope('l2normalize'):
        return v / (tf.reduce_sum(v ** 2) ** 0.5 + eps)

def conv2d(x, W, kernel_stride):
    # API: must strides[0]=strides[4]=1
    # return tf.nn.atrous_conv2d(x,W,rate=2,padding='SAME')
    return tf.nn.conv2d(x, W, strides=[1, kernel_stride, kernel_stride, 1], padding='SAME')

def apply_max_pooling(x, pooling_height, pooling_width, pooling_stride):
    # API: must ksize[0]=ksize[4]=1, strides[0]=strides[4]=1
    return tf.nn.max_pool(x, ksize=[1, pooling_height, pooling_width, 1],
                          strides=[1, pooling_stride, pooling_stride, 1], padding='SAME')

def apply_conv2d(x, filter_height, filter_width, out_channels, kernel_stride,spectral_normed= False, name='conv'):
    with tf.variable_scope(name):
        weight=tf.get_variable('weight',shape=[filter_height, filter_width, x.get_shape()[-1], out_channels],
                               initializer=tf.random_normal_initializer(stddev=0.1))
        if spectral_normed:
            weight = spectral_normed_weight(weight)
        variable_summary(weight)
        tf.add_to_collection('regularizer',spectral_norm(weight))
        bias = tf.get_variable('bias',[out_channels],initializer=tf.constant_initializer(0.1))  # each feature map shares the same weight and bias
        variable_summary(bias)
        conv=tf.add(conv2d(x,weight,kernel_stride),bias,name='conv_add_b')
        return tf.nn.selu(conv)

def apply_fully_connect(x, x_size, fc_size,spectral_normed= False, name='fully_connect'):
    with tf.variable_scope(name):
        fc_weight = tf.get_variable('fc_weight',[x_size, fc_size],
                                    initializer=tf.random_normal_initializer(stddev=0.1))
        if spectral_normed:
            fc_weight = spectral_normed_weight(fc_weight)
        variable_summary(fc_weight)
        tf.add_to_collection('regularizer', spectral_norm(fc_weight))
        mul = tf.matmul(x, fc_weight, name='fc_mul')
        fc_bias = tf.get_variable('fc_bias',[fc_size],initializer=tf.constant_initializer(0.1))
        variable_summary(fc_bias)
        fc_mul = tf.add(mul, fc_bias, name="mul_add_b")
        return tf.nn.selu(fc_mul)

def apply_readout(x, x_size, readout_size,spectral_normed= False,name='readout'):
    with tf.variable_scope(name):
        readout_weight = tf.get_variable('readout_weight',[x_size, readout_size],
                                         initializer=tf.random_normal_initializer(stddev=0.1))
        if spectral_normed:
            readout_weight = spectral_normed_weight(readout_weight)
        variable_summary(readout_weight)
        tf.add_to_collection('regularizer', spectral_norm(readout_weight))
        mul = tf.matmul(x, readout_weight, name="read_mul")
        readout_bias = tf.get_variable('readout_bias',[readout_size],initializer=tf.constant_initializer(0.1))
        variable_summary(readout_bias)
        readout_mul = tf.add(mul, readout_bias, name="readout_add_b")
        return readout_mul

print("\n**********(" + time.asctime(time.localtime(time.time())) + ") Define parameters and functions End **********")
print("\n**********(" + time.asctime(time.localtime(time.time())) + ") Define NN structure Begin: **********")

# input placeholder
cnn_in = tf.placeholder(tf.float32, shape=[None, input_height, input_width, input_channel_num], name='cnn_in')
Y = tf.placeholder(tf.float32, shape=[None, n_labels], name='Y')
keep_prob = tf.placeholder(tf.float32, name='keep_prob')
phase_train = tf.placeholder(tf.bool, name='phase_train')
###########################################################################################
# add cnn parallel to network
###########################################################################################
conv_1 = apply_conv2d(cnn_in, kernel_height_1st, kernel_width_1st, conv_channel_num, kernel_stride,name='conv1')
conv_2 = apply_conv2d(conv_1, kernel_height_2nd, kernel_width_2nd, conv_channel_num * 2,kernel_stride,name='conv2')
# conv_2 = apply_max_pooling(conv_1, pooling_height, pooling_width, pooling_stride)
conv_3 = apply_conv2d(conv_2, kernel_height_3rd, kernel_width_3rd, conv_channel_num * 4, kernel_stride, name='conv3')
# conv_3 = apply_conv2d(conv_2, kernel_height_3rd, kernel_width_3rd, conv_channel_num * 2, kernel_stride, name='conv3')
conv_4= apply_max_pooling(conv_3, pooling_height, pooling_width, pooling_stride)
# conv_4 = apply_conv2d(conv_3, kernel_height_4th, kernel_width_4th, conv_channel_num, kernel_stride, name='conv4')

shape = conv_4.get_shape().as_list()
conv_4_flat = tf.reshape(conv_4, [-1, shape[1] * shape[2] * shape[3]])
cnn_fc = apply_fully_connect(conv_4_flat, shape[1] * shape[2] * shape[3], fc_size, name="fc")
cnn_fc_drop = tf.nn.dropout(cnn_fc, keep_prob)
y_ = apply_readout(cnn_fc_drop, fc_size, n_labels, name='readout')
y_pred = tf.argmax(tf.nn.softmax(y_), 1, name="y_pred")
y_posi = tf.nn.softmax(y_, name="y_posi")

# l2 regularization
l2 = lambda_loss_amount * sum(
    tf.nn.l2_loss(tf_var) for tf_var in tf.trainable_variables()
)
regular = lambda_loss_amount * (tf.add_n(tf.get_collection('regularizer')))

if enable_penalty:
    # cross entropy cost function
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y_, labels=Y)+ l2 , name='loss')
else:
    # cross entropy cost function
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y_, labels=Y), name='loss')

optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)
# get correctly predicted object and accuracy
correct_prediction = tf.equal(tf.argmax(tf.nn.softmax(y_), 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name='accuracy')

# Merge all summaries into a single op
merged_summary_op = tf.summary.merge_all()
saver = tf.train.Saver()

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
    fold_size = datasets.shape[0] // fold
    indexes_list = [i for i in range(len(datasets))]
    indexes = np.array(indexes_list)
    split_list = [i for i in range(curr_fold * fold_size, (curr_fold + 1) * fold_size)]
    split = np.array(split_list)
    cnn_test_x = datasets[split]
    test_y = labels[split]

    split = np.array(list(set(indexes_list) ^ set(split_list)))
    cnn_train_x = datasets[split]
    train_y = labels[split]
    train_sample = train_y.shape[0]

    # shuffle data
    index = np.array(range(0, len(train_y)))
    np.random.shuffle(index)

    cnn_train_x = cnn_train_x[index]
    train_y = train_y[index]

    print("training examples:", train_sample)
    test_sample = test_y.shape[0]
    print("test examples    :", test_sample)
    # set train batch number per epoch
    batch_num_per_epoch = math.floor(cnn_train_x.shape[0] / batch_size) + 1
    # set test batch number per epoch
    accuracy_batch_size = batch_size
    train_accuracy_batch_num = batch_num_per_epoch
    test_accuracy_batch_num = math.floor(cnn_test_x.shape[0] / batch_size) + 1

    with tf.Session(config=config) as session:

        # writer = tf.summary.FileWriter("F:/CB/My_experiment_Deap/final_cnnModel_log", session.graph)
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
                cnn_batch = cnn_train_x[start:(start + offset), :, :, :]
                cnn_batch = cnn_batch.reshape(len(cnn_batch) * window_size, input_height, input_width,
                                              input_channel_num)
                # print("cnn_batch shape:",cnn_batch.shape)
                batch_y = train_y[start:(offset + start), :]
                _, c = session.run([optimizer, cost ],
                                   feed_dict={cnn_in: cnn_batch, Y: batch_y, keep_prob: 1 - dropout_prob,
                                              phase_train: True})
                # summary, _ = session.run([merged_summary_op, optimizer], feed_dict={cnn_in: cnn_batch, Y: batch_y, keep_prob: 1 - dropout_prob,
                #                               phase_train: True})
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
                    train_cnn_batch = cnn_train_x[start:(start + offset), :, :, :]
                    train_cnn_batch = train_cnn_batch.reshape(len(train_cnn_batch) * window_size, input_height,
                                                              input_width, input_channel_num)
                    train_batch_y = train_y[start:(start + offset), :]

                    train_a, train_c = session.run([accuracy, cost],
                                                   feed_dict={cnn_in: train_cnn_batch, Y: train_batch_y, keep_prob: 1.0,
                                                              phase_train: False})

                    # saver.save(session, "model/my_cnn/%s_%s_model"% (sub,arousal_or_valence), global_step=10)
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
                    test_cnn_batch = cnn_test_x[start:(offset + start), :, :, :]
                    test_cnn_batch = test_cnn_batch.reshape(len(test_cnn_batch) * window_size, input_height,
                                                            input_width, input_channel_num)
                    test_batch_y = test_y[start:(offset + start), :]

                    test_a, test_c = session.run([accuracy, cost],
                                                 feed_dict={cnn_in: test_cnn_batch, Y: test_batch_y, keep_prob: 1.0,
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
            cnn_train_x = cnn_train_x[index]
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
            test_cnn_batch = cnn_test_x[start:(offset + start), :, :, :]
            test_cnn_batch = test_cnn_batch.reshape(len(test_cnn_batch) * window_size, input_height, input_width,
                                                    input_channel_num)
            test_batch_y = test_y[start:(offset + start), :]

            test_a, test_c, test_p, test_r = session.run([accuracy, cost, y_pred, y_posi ],
                                                         feed_dict={cnn_in: test_cnn_batch, Y: test_batch_y,
                                                                    keep_prob: 1.0, phase_train: False})
            test_t = test_batch_y

            test_accuracy = np.append(test_accuracy, test_a)
            test_loss = np.append(test_loss, test_c)
            test_pred = np.append(test_pred, test_p)
            test_true = np.vstack([test_true, test_t])
            test_posi = np.vstack([test_posi, test_r])

        test_pred_1_hot = np.asarray(pd.get_dummies(test_pred), dtype=np.int8)
        test_true_list = tf.argmax(test_true, 1).eval()
        # recall
        test_recall = recall_score(test_true, test_pred_1_hot, average = None)
        # precision
        test_precision = precision_score(test_true, test_pred_1_hot, average = None)
        # f1 score
        test_f1 = f1_score(test_true, test_pred_1_hot, average = None)
        # fpr, tpr, auc
        fpr = {}
        tpr = {}
        roc_auc = []
        i = 0
        for key in one_hot_labels:
            fpr[key], tpr[key], _ = roc_curve(test_true[:, i], test_posi[:, i])
            roc_auc.append(auc(fpr[key], tpr[key]))
            i += 1
        # test_auc = calcAUC_byProb(test_true_list, np.mean(test_posi, axis=1))

        # confusion matrix
        # confusion_matrix = confusion_matrix(test_true_list, test_pred)
        print("********************recall:", test_recall)
        print("*****************precision:", test_precision)
        print("******************f1_score:", test_f1)
        print("**********confusion_matrix:\n", confusion_matrix)

        print("(" + time.asctime(time.localtime(time.time())) + ") Final Test Cost: ", np.mean(test_loss),
              "Final Test Accuracy: ", np.mean(test_accuracy), "recall", np.mean(test_recall), "precision",
              np.mean(test_precision),
              "f1", np.mean(test_f1), "auc", np.mean(roc_auc))

    sum_acc += np.mean(test_accuracy)
    sum_recall += np.mean(test_recall)
    sum_precision += np.mean(test_precision)
    sum_f1 += np.mean(test_f1)
    sum_auc += np.mean(roc_auc)

mean_acc = sum_acc / fold * 100
mean_recall = sum_recall / fold
mean_precision = sum_precision / fold
mean_f1 = sum_f1 / fold
mean_auc = sum_auc / fold

files = file_name()
save_file_name = arousal_or_valence + "_" + str(bands) + ".xls"
print("acc:", mean_acc, "recall:", mean_recall, "precision:", mean_precision, "f1_score:", mean_f1, "auc:",
      mean_auc, "save path: D:/CB/deapProject/result/cnn_result/subIndependent_result/", save_file_name)
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
    book.save("D:/CB/deapProject/result/cnn_result/subIndependent_result/" + save_file_name)

rexcel = open_workbook(
    "D:/CB/deapProject/result/cnn_result/subIndependent_result/" + save_file_name)  # 用wlrd提供的方法读取一个excel文件
excel = copy(rexcel)  # 用xlutils提供的copy方法将xlrd的对象转化为xlwt的对象
table = excel.get_sheet(0)  # 用xlwt对象的方法获得要操作的sheet

table.write(1, 1,  mean_acc)
table.write(1, 2,  mean_recall)
table.write(1, 3,  mean_precision)
table.write(1, 4,  mean_f1)
table.write(1, 5,  mean_auc)
excel.save("D:/CB/deapProject/result/cnn_result/subIndependent_result/" + save_file_name)
