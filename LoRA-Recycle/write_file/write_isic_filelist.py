import numpy as np
import csv
import os
import pandas as pd
DATA_PATH=''
SPLIT_PATH = ''
assert len(DATA_PATH)!=0,'You should input the data_path!'
assert len(SPLIT_PATH)!=0,'You should input the savedir!'
os.makedirs(SPLIT_PATH, exist_ok=True)
split_list = ['meta_train', 'meta_val', 'meta_test']

data_info = pd.read_csv('Path_to_ISIC2018_Task3_Training_GroundTruth.csv', skiprows=[0], header=None)
# First column contains the image paths
image_name = [os.path.join(DATA_PATH,i)+'.jpg' for i in np.asarray(data_info.iloc[:, 0])]

labels = np.asarray(data_info.iloc[:, 1:])
labels = [str(i) for i in (labels!=0).argmax(axis=1)]

for split in ['meta_train']:
    fo = open(SPLIT_PATH + split + ".csv", "w",newline='')
    writer = csv.writer(fo)
    writer.writerow(['filename','label'])
    path_label=list(zip(image_name,labels))
    path_label.sort(key=lambda a: a[1])
    print(np.array(path_label))
    writer.writerows(np.array(path_label))
    fo.close()
    print("%s -OK" %split)