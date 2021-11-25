import os
import sys
import math
import numpy as np
import pandas as pd
import scipy.io as sio
from sklearn import preprocessing
from scipy.signal import butter, lfilter
import seaborn as sn
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y

def read_file(file):
    data = sio.loadmat(file)
    data = data['data']
    # print(data.shape)
    return data

def compute_DE(signal):
    variance = np.var(signal,ddof=1)
    return math.log(2*math.pi*math.e*variance)/2
#将信号分解到4个频带上
def decompose(file):
    # trial*channel*sample
    start_index = 384 #3s pre-trial signals
    data = read_file(file)
    frequency = 128
    #1次实验信号分解之后的格式
    decomposed_de = np.empty([0,4,60])
    base_DE = np.empty([0,128])

    for trial in range(40):
        temp_base_DE = np.empty([0])
        temp_base_theta_DE = np.empty([0])
        temp_base_alpha_DE = np.empty([0])
        temp_base_beta_DE = np.empty([0])
        temp_base_gamma_DE = np.empty([0])
        temp_de = np.empty([0,60])

        for channel in range(32):
            trial_signal = data[trial,channel,384:]
            base_signal = data[trial,channel,:384]
            #****************compute base DE****************
            base_theta = butter_bandpass_filter(base_signal, 4, 8, frequency, order=3)
            base_alpha = butter_bandpass_filter(base_signal, 8,14, frequency, order=3)
            base_beta = butter_bandpass_filter(base_signal,14,31, frequency, order=3)
            base_gamma = butter_bandpass_filter(base_signal,31,45, frequency, order=3)

            base_theta_DE = (compute_DE(base_theta[:128]) + compute_DE(base_theta[128:256]) + compute_DE(base_theta[256:])) / 3
            base_alpha_DE = (compute_DE(base_alpha[:128]) + compute_DE(base_alpha[128:256]) + compute_DE(base_alpha[256:])) / 3
            base_beta_DE = (compute_DE(base_beta[:128]) + compute_DE(base_beta[128:256]) + compute_DE(base_beta[256:])) / 3
            base_gamma_DE = (compute_DE(base_gamma[:128]) + compute_DE(base_gamma[128:256]) + compute_DE(base_gamma[256:])) / 3

            temp_base_theta_DE = np.append(temp_base_theta_DE,base_theta_DE)
            temp_base_gamma_DE = np.append(temp_base_gamma_DE,base_gamma_DE)
            temp_base_beta_DE = np.append(temp_base_beta_DE,base_beta_DE)
            temp_base_alpha_DE = np.append(temp_base_alpha_DE,base_alpha_DE)

            theta = butter_bandpass_filter(trial_signal, 4, 8, frequency, order=3)
            alpha = butter_bandpass_filter(trial_signal, 8, 14, frequency, order=3)
            beta = butter_bandpass_filter(trial_signal, 14, 31, frequency, order=3)
            gamma = butter_bandpass_filter(trial_signal, 31, 45, frequency, order=3)

            DE_theta = np.zeros(shape=[0],dtype = float)
            DE_alpha = np.zeros(shape=[0],dtype = float)
            DE_beta =  np.zeros(shape=[0],dtype = float)
            DE_gamma = np.zeros(shape=[0],dtype = float)

            for index in range(60):
                DE_theta =np.append(DE_theta,compute_DE(theta[index*frequency:(index+1)*frequency]))
                DE_alpha =np.append(DE_alpha,compute_DE(alpha[index*frequency:(index+1)*frequency]))
                DE_beta =np.append(DE_beta,compute_DE(beta[index*frequency:(index+1)*frequency]))
                DE_gamma =np.append(DE_gamma,compute_DE(gamma[index*frequency:(index+1)*frequency]))
            temp_de = np.vstack([temp_de,DE_theta])
            temp_de = np.vstack([temp_de,DE_alpha])
            temp_de = np.vstack([temp_de,DE_beta])
            temp_de = np.vstack([temp_de,DE_gamma])
        temp_trial_de = temp_de.reshape(-1,4,60)
        decomposed_de = np.vstack([decomposed_de,temp_trial_de])

        temp_base_DE = np.append(temp_base_theta_DE,temp_base_alpha_DE)
        temp_base_DE = np.append(temp_base_DE,temp_base_beta_DE)
        temp_base_DE = np.append(temp_base_DE,temp_base_gamma_DE)
        base_DE = np.vstack([base_DE,temp_base_DE])
        #transpose()表示将1轴和3轴进行互换
    decomposed_de = decomposed_de.reshape(-1,32,4,60).transpose([0,3,2,1]).reshape(-1,4,32).reshape(-1,128)
    print(decomposed_de.shape)
    print("base_DE shape:",base_DE.shape)
    print("trial_DE shape:",decomposed_de.shape)
    return base_DE,decomposed_de

def get_labels(file):
    #0 valence, 1 arousal, 2 dominance, 3 liking
    valence_labels = sio.loadmat(file)["labels"][:,0] > 5	# valence labels
    arousal_labels = sio.loadmat(file)["labels"][:,1] > 5	# arousal labels
    dominance_labels=sio.loadmat(file)["labels"][:,2] > 5 #dominance labels
    liking_labels=sio.loadmat(file)["labels"][:,3] > 5 #liking labels
    op_labels = []
    final_valence_labels = np.empty([0])
    final_arousal_labels = np.empty([0])
    final_dominance_labels=np.empty([0])
    final_liking_labels=np.empty([0])
    final_op_labels=np.empty([0])
    for i in range(len(valence_labels)):
        if sio.loadmat(file)["labels"][:,0][i]>5 and sio.loadmat(file)["labels"][:,1][i]>5:
            op=0
        elif sio.loadmat(file)["labels"][:,0][i]<5 and sio.loadmat(file)["labels"][:,1][i]>5:
            op=1
        elif sio.loadmat(file)["labels"][:,0][i]<5 and sio.loadmat(file)["labels"][:,1][i]<5:
            op=2
        else:
            op=3
        op_labels=np.append(op_labels,op)
        for j in range(0, 60):
            final_valence_labels = np.append(final_valence_labels,valence_labels[i])
            final_arousal_labels = np.append(final_arousal_labels,arousal_labels[i])
            final_dominance_labels=np.append(final_dominance_labels,dominance_labels[i])
            final_liking_labels=np.append(final_liking_labels,liking_labels[i])
            final_op_labels=np.append(final_op_labels,op_labels[i])
    print("labels:",final_arousal_labels.shape)
    return final_arousal_labels,final_valence_labels,final_dominance_labels,final_liking_labels,final_op_labels

def get_vector_deviation(vector1, vector2):
    return vector1 - vector2

def get_dataset_deviation(trial_data, base_data):
    new_dataset = np.empty([0, 128])  # 在1s的时间段有128个数据点
    for i in range(0, 2400):  # 共分成了800段
        base_index = i // 60  # 对20进行整除
        # print(base_index)
        base_index = 39 if base_index == 40 else base_index
        new_record = get_vector_deviation(trial_data[i], base_data[base_index]).reshape(1, 128)
        # print(new_record.shape)
        new_dataset = np.vstack([new_dataset, new_record])
    # print("new shape:",new_dataset.shape)
    return new_dataset

#得到的1维微分熵特征就是trial_data是800*128（800段，每一段4*32表示4个频带，32个通道）一行表示一段中的4个频带上的32个DE
if __name__ == '__main__':
    dataset_dir = "F:/DEAP数据集(EEG)/data_preprocessed_matlab/"

    result_dir = "D:/CB/deapProject/DE_feature/1D_DE_noNorm/"
    if os.path.isdir(result_dir)==False:
        os.makedirs(result_dir)

    for file in os.listdir(dataset_dir):
        print("processing: ",file,"......")
        file_path = os.path.join(dataset_dir,file)
        base_DE,trial_DE = decompose(file_path)
        data = get_dataset_deviation(trial_DE, base_DE)
        arousal_labels,valence_labels,dominance_labels,liking_labels,op_labels= get_labels(file_path)
        sio.savemat(result_dir+"DE_"+file,{"data":data,"valence_labels":valence_labels,"arousal_labels":arousal_labels,
                                           "dominance_labels": dominance_labels, "liking_labels": liking_labels, "op_labels": op_labels})

        print(trial_DE.shape)

        trial_DE[2] = preprocessing.scale(trial_DE[2], axis=0, with_mean=True, with_std=True, copy=True)
        trial_DE[6] = preprocessing.scale(trial_DE[6], axis=0, with_mean=True, with_std=True, copy=True)
        trial_DE[10] = preprocessing.scale(trial_DE[10], axis=0, with_mean=True, with_std=True, copy=True)
        trial_DE[19] = preprocessing.scale(trial_DE[19], axis=0, with_mean=True, with_std=True, copy=True)
        print(trial_DE[2].shape)
        print(trial_DE[6].shape)

        f, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(figsize=(10, 8), nrows=5)
        rx_tick = np.arange(0, 2500, 500)
        ry_tick = np.arange(1, 5, 1)
        sn.heatmap(np.array(trial_DE[2][:,np.newaxis]), ax=ax1, cmap='jet', xticklabels=rx_tick, cbar=False)
        ax1.set_ylabel('F7')
        ax1.set_xticks(rx_tick)
        ax1.set_xticklabels(rx_tick, rotation='horizontal')
        ax1.set_yticklabels(ry_tick, rotation='horizontal')

        sn.heatmap(np.array(trial_DE[6][:, np.newaxis]), ax=ax2, cmap='jet', xticklabels=rx_tick, cbar=False)
        ax2.set_ylabel('T7')
        ax2.set_xticks(rx_tick)
        ax2.set_xticklabels(rx_tick, rotation='horizontal')
        ax2.set_yticklabels(ry_tick, rotation='horizontal')

        sn.heatmap(np.array(trial_DE[10][:, np.newaxis]), ax=ax3, cmap='jet', xticklabels=rx_tick, cbar=False)
        ax3.set_ylabel('P7')
        ax3.set_xticks(rx_tick)
        ax3.set_xticklabels(rx_tick, rotation='horizontal')
        ax3.set_yticklabels(ry_tick, rotation='horizontal')

        sn.heatmap(np.array(trial_DE[19][:, np.newaxis]), ax=ax4, cmap='jet', xticklabels=rx_tick, cbar=False)
        ax4.set_ylabel('P8')
        ax4.set_xticks(rx_tick)
        ax4.set_xticklabels(rx_tick, rotation='horizontal')
        ax4.set_yticklabels(ry_tick, rotation='horizontal')

        x = [i for i in range(2400)]
        ax5 = plt.plot(x,valence_labels, color='b', linewidth = 2)
        ax5 = plt.plot(x,arousal_labels, color='r', linewidth = 2)
        plt.xlabel('Time[/s]')
        plt.ylabel('Label')
        plt.ylim(-0.1,1.1)
        plt.xlim(-0.1,2400)
        ax = plt.gca()
        x_major_locator = MultipleLocator(500)
        ax.xaxis.set_major_locator(x_major_locator)
        y_major_locator = MultipleLocator(1)
        ax.yaxis.set_major_locator(y_major_locator)
        plt.tight_layout()
        f.savefig('heatmap1.jpg', dpi=500)

        # x = [i for i in range(2400)]
        # y1 = valence_labels
        # plt.figure(figsize=(15, 6))
        # plt.plot(x, y1, label='valence', color='b', linewidth=2)
        # # ax5.set_ylabel('Label')
        # # ax5.set_xlabel('Time[/s]')
        # plt.xlabel('Time[/s]')
        # plt.ylabel('Label')
        # ax = plt.gca()
        # y_major_locator = MultipleLocator(1)
        # ax.yaxis.set_major_locator(y_major_locator)
        # plt.ylim(-0.1, 1.1)
        # plt.legend()
        # plt.savefig('label.jpg',dpi=500)
        # plt.show()

        # # plt.subplot(2,1,1)
        # plt.plot(x,y1,label='valence', color='b',linewidth = 2)
        # # plt.subplot(2, 1, 1)
        # # plt.plot(x,y2,label='arousal',linewidth=3)
        # plt.xlabel('Time[/s]')
        # plt.ylabel('Label')
        # y_major_locator=MultipleLocator(1)
        # ax=plt.gca()
        # #ax为两条坐标轴的实例
        # ax.yaxis.set_major_locator(y_major_locator)
        # plt.ylim(-0.1,1.1)
        # plt.legend()
        # plt.savefig('label.jpg',dpi=500)
        # plt.show()


