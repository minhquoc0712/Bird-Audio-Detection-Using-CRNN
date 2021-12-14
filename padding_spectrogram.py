import numpy as np
from os import listdir
from os.path import isfile, join
import csv


path1 = 'output_mel\\'
file_name = [f for f in listdir(path1) if isfile(join(path1, f))]

label_file = ['warblrb10k_public_metadata_2018.csv', 'ff1010bird_metadata_2018.csv']
label_dict = {}

with open(label_file[0], "r") as f:
    reader = csv.reader(f, delimiter=",")
    for i, line in enumerate(reader):
        if i == 0:
            continue
        label_dict[line[0]] = int(line[2])

with open(label_file[1], "r") as f:
    reader = csv.reader(f, delimiter=",")
    for i, line in enumerate(reader):
        if i == 0:
            continue
        label_dict[line[0]] = int(line[2])

length = np.zeros((len(file_name), 1))
for i in range(len(file_name)):
    x = np.load(path1 + file_name[i])
    print(i)
    if x.shape[1] <= 865:
        x = np.concatenate([x, -15.00 * np.ones((40, 865 - x.shape[1]))], axis=1)
        filename = file_name[i].split('.')
        filename = filename[0]
        print(x.shape)

        if label_dict[filename] == 1:
            np.save('presence\\' + filename, x)
        else:
            np.save('absence\\' + filename, x)


