import random

import numpy as np
from os import listdir
from os.path import isfile, isdir, join
import csv
import os
DATA_PATH=''
SPLIT_PATH = ''
assert len(DATA_PATH)!=0,'You should input the data_path!'
assert len(SPLIT_PATH)!=0,'You should input the savedir!'
os.makedirs(SPLIT_PATH, exist_ok=True)
split_list = ['meta_train', 'meta_val', 'meta_test']

SPLIT={
'meta_train': ['AnnualCrop' , 'Forest' , 'HerbaceousVegetation' , 'Highway' , 'Industrial' , 'Pasture' , 'PermanentCrop',  'Residential' , 'River' , 'SeaLake'],
    'meta_val': [],
    'meta_test': [],
}

folder_list = [f for f in listdir(DATA_PATH) if isdir(join(DATA_PATH, f))]
folder_list.sort()
label_dict = dict(zip(range(0,len(folder_list)),folder_list))

classfile_list_all = []

for i, folder in enumerate(folder_list):
    folder_path = join(DATA_PATH, folder)
    classfile_list_all.append( [ join(folder_path, cf) for cf in listdir(folder_path) if (isfile(join(folder_path,cf)) and cf[0] != '.' )])
    random.shuffle(classfile_list_all[i])


for split in split_list:
    num=0
    file_list = []
    label_list = []
    for i, classfile_list in enumerate(classfile_list_all):
        if label_dict[i] in SPLIT[split]:
            file_list = file_list + classfile_list
            label_list = label_list + np.repeat(label_dict[i], len(classfile_list)).tolist()
            num = num + 1
    print('split_num:',num)
    fo = open(SPLIT_PATH + split + ".csv", "w",newline='')
    writer = csv.writer(fo)
    writer.writerow(['filename','label'])
    temp=np.array(list(zip(file_list,label_list)))
    writer.writerows(temp)
    fo.close()
    print("%s -OK" %split)