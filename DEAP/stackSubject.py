import sys
import os
import numpy as np
import scipy.io as sio
import pandas as pd

data_all=np.empty([0,128])
valence_all=np.empty([0])
arousal_all=np.empty([0])
dominance_all=np.empty([0])
liking_all=np.empty([0])
op_all=np.empty([0])
for sub_id in range(1,33):
    print("processing ",sub_id)
    sub_id = "%02d" % sub_id
    with_DE = sio.loadmat("D:/CB/deapProject/DE_feature/1D_DE_with/DE_s"+str(sub_id)+".mat")
    with_DE_data = with_DE["data"]
    with_valence=np.squeeze(with_DE['valence_labels'])
    with_arousal=np.squeeze(with_DE['arousal_labels'])
    with_dominance=np.squeeze(with_DE['dominance_labels'])
    with_liking=np.squeeze(with_DE['liking_labels'])
    with_op=np.squeeze(with_DE['op_labels'])

    data_all=np.vstack((data_all,with_DE_data))
    valence_all=np.hstack((valence_all,with_valence))
    arousal_all = np.hstack((arousal_all,with_arousal))
    dominance_all = np.hstack((dominance_all,with_dominance))
    liking_all = np.hstack((liking_all,with_liking))
    op_all = np.hstack((op_all,with_op))
print(data_all.shape)
print(valence_all.shape)

result_dir = "D:/CB/deapProject/DE_feature/DE_feature_allSub/"
if os.path.isdir(result_dir) == False:
    os.makedirs(result_dir)
sio.savemat(result_dir + "DE_allSub",
            {"data": data_all, "valence_labels": valence_all, "arousal_labels": arousal_all,
             "dominance_labels": dominance_all, "liking_labels": liking_all, "op_labels": op_all})




#
# result_dir = "D:/CB/My_experiment_Deap/dataset_feature/3D_dataset/3D_temporal_with/"
# if os.path.isdir(result_dir)==False:
#     os.makedirs(result_dir)
# sio.savemat(result_dir+"std_"+sub_id,{"data":final_data,"valence_labels":valence_labels,"arousal_labels":arousal_labels})



