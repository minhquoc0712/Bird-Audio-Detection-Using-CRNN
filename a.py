import numpy as np
from os import listdir
from os.path import isfile, join
from tensorflow.keras.models import load_model

file_test = [f for f in listdir('test_mel_865\\') if isfile(join('test_mel_865\\', f))]

X_test = np.zeros((len(file_test), 865, 40, 1))

list = []
for i in range(len(file_test)):
    file_name = file_test[i].split('.')
    list.append(file_name[0])
    X_test[i, :, :, :] = np.expand_dims(np.load('test_mel_865\\' + file_test[i]).T, axis=2)

model = load_model('best_model_865.hdf5')
y_pred = model.predict(X_test, batch_size=128)

text_file = open('result_865_6epoch.txt', 'w')
text_file.write('ID,Predicted\n')
for i in range(y_pred.shape[0]):
    print(i)
    text_file.write(f"{list[i]},{y_pred[i, 0]}\n")
text_file.close()