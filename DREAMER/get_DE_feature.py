import os
import math
import numpy as np
import scipy.io as sio
from scipy.signal import butter, lfilter

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
    data=sio.loadmat(file)
    data = data['struct']
    # print(data.shape)
    return data

def compute_DE(signal):
    variance = np.var(signal,ddof=1)
    return math.log(2*math.pi*math.e*variance) / 2

def decompose(file):
    # trial*channel*sample
    data = read_file(file)
    frequency = 128

    decomposed_de = np.empty([0,4,60])

    for trial in range(18):
        trial_signal = np.transpose(data[0,0][0,trial])
        temp_de = np.empty([0,60])

        for channel in range(14):
            channel_signal=trial_signal[channel]
            theta = butter_bandpass_filter(channel_signal, 4, 8, frequency, order=3)
            alpha = butter_bandpass_filter(channel_signal, 8, 14, frequency, order=3)
            beta = butter_bandpass_filter(channel_signal, 14, 31, frequency, order=3)
            gamma = butter_bandpass_filter(channel_signal, 31, 45, frequency, order=3)

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

    decomposed_de = decomposed_de.reshape(-1,14,4,60).transpose([0,3,2,1]).reshape(-1,4,14).reshape(-1,56)
    print("trial_DE shape:",decomposed_de.shape)
    return decomposed_de

def get_labels(file):
    #0 valence, 1 arousal, 2 dominance
    valence_labels = read_file(file)[0,1][0] >= 3	# valence labels
    arousal_labels = read_file(file)[0,2][0] >= 3	# arousal labels
    dominance_labels = read_file(file)[0,3][0] >= 3 #dominance labels
    # liking_labels=sio.loadmat(file)["labels"][:,3]>5 #liking labels
    op_labels = []
    final_valence_labels = np.empty([0])
    final_arousal_labels = np.empty([0])
    final_dominance_labels=np.empty([0])
    # final_liking_labels=np.empty([0])
    final_op_labels=np.empty([0])
    for i in range(len(valence_labels)):
        if read_file(file)[0,1][0][i] >= 3 and read_file(file)[0,2][0][i] >= 3:
            op=0
        elif read_file(file)[0,1][0][i] < 3 and read_file(file)[0,2][0][i] >= 3:
            op=1
        elif read_file(file)[0,1][0][i] < 3 and read_file(file)[0,2][0][i] < 3:
            op=2
        else:
            op=3
        op_labels=np.append(op_labels,op)
        for j in range(0,60):
            final_valence_labels = np.append(final_valence_labels,valence_labels[i])
            final_arousal_labels = np.append(final_arousal_labels,arousal_labels[i])
            final_dominance_labels=np.append(final_dominance_labels,dominance_labels[i])
            # final_liking_labels=np.append(final_liking_labels,liking_labels[i])
            final_op_labels=np.append(final_op_labels,op_labels[i])
    print("labels:",final_arousal_labels.shape)
    # return final_arousal_labels,final_valence_labels,final_dominance_labels,final_liking_labels,final_op_labels
    return final_arousal_labels,final_valence_labels,final_dominance_labels,final_op_labels

def wgn(x, snr):
    snr = 10**(snr/10.0)
    xpower = np.sum(x**2)/len(x)
    npower = xpower / snr
    return np.random.randn(len(x)) * np.sqrt(npower)

def feature_normalize(data):
    mean = data[data.nonzero()].mean()
    sigma = data[data. nonzero ()].std()
    data_normalized = data
    data_normalized[data_normalized.nonzero()] = (data_normalized[data_normalized.nonzero()] - mean)/sigma
    return data_normalized


if __name__ == '__main__':
    dataset_dir = "D:/CB/dreamer_project/subject_separated_removeBase/"

    result_dir = "D:/CB/dreamer_project/dreamer_feature/DE_feature1/"
    if os.path.isdir(result_dir)==False:
        os.makedirs(result_dir)

    for file in os.listdir(dataset_dir):
        print("processing: ",file,"......")
        file_path = os.path.join(dataset_dir,file)
        trial_DE = decompose(file_path)
        arousal_labels, valence_labels, dominance_labels, op_labels = get_labels(file_path)
        sio.savemat(result_dir+"DE_"+file,{"data":trial_DE,"valence_labels":valence_labels,"arousal_labels":arousal_labels,
                                           "dominance_labels":dominance_labels,"op_labels": op_labels})
