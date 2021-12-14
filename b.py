import numpy as np
import librosa as lb
from librosa.display import specshow
import matplotlib.pyplot as plt
from os import listdir
from os.path import isfile, join

path1 = 'warblrb10k_public_wav\wav\\'
onlyfiles1 = [f for f in listdir(path1) if isfile(join(path1, f))]
path2 = 'ff1010bird_wav\wav\\'
onlyfiles2 = [f for f in listdir(path1) if isfile(join(path2, f))]

length = np.zeros((8000, 1))
for i in range(8000):
    i
    y, sr = lb.load(path2 + onlyfiles2[i], sr=44100)
    length[i] = y.shape[0]

plt.figure()
plt.hist(length, bins='auto')
plt.show()