import numpy as np
import librosa as lb
from librosa.display import specshow
import matplotlib.pyplot as plt
from os import listdir
from os.path import isfile, join

input_path = 'input_file\\'
output_path = 'output_mel\\'
file_name = [f for f in listdir(input_path) if isfile(join(input_path, f))]

length = np.zeros((len(file_name), 1))
kwargs_for_mel = {'n_mels': 40}
for i in range(len(file_name)):

    y, sr = lb.load(input_path + file_name[i], sr=44100)
    y = y * 1 / np.max(np.abs(y))
    x = lb.feature.melspectrogram(y, sr=44100, n_fft=1024, hop_length=512,
                                  window='hamm', power=1, **kwargs_for_mel)

    x = np.log(x)

    filename = file_name[i].split('.')[0]
    length[i] = x.shape[1]
    print(x.shape)
    np.save(output_path + filename, x)


plt.figure()
plt.hist(length, bins='auto')
plt.show()

