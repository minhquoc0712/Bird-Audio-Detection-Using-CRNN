import numpy as np
import librosa as lb
from librosa.display import specshow
import matplotlib.pyplot as plt

from os import listdir
from os.path import isfile, join

S = np.load('output_mel/00a29637-99aa-4f23-97c9.npy')

plt.figure(figsize=(10, 4))
specshow(S, x_axis='time', y_axis='linear', sr=44100)
plt.colorbar(format='%+2.0f dB')
plt.title('Mel-frequency spectrogram')
plt.tight_layout()
plt.show()