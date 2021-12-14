import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, GRU, Dense, Flatten,\
        BatchNormalization, Dropout, Reshape
from tensorflow.keras.optimizers.schedules import ExponentialDecay
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import binary_crossentropy
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.metrics import AUC
import numpy as np
from os import listdir
from os.path import isfile, join
from tensorflow.keras.models import load_model


files_absence = [f for f in listdir('absence\\') if isfile(join('absence\\', f))]
files_presence = [f for f in listdir('presence\\') if isfile(join('presence\\', f))]

X = np.zeros((len(files_absence) + len(files_presence), 865, 40, 1))

for i in range(len(files_absence)):
    X[i, :, :, :] = np.expand_dims(np.load('absence\\' + files_absence[i]).T, axis=2)

for i in range(len(files_presence)):
    X[i + len(files_absence), :, :, :] = np.expand_dims(np.load('presence\\' + files_presence[i]).T, axis=2)

y = np.concatenate([np.zeros((len(files_absence), 1)), np.ones((len(files_presence), 1))])

print(X.shape)
print(y.shape)

ind_perm = np.random.permutation(X.shape[0])
X = X[ind_perm, :, :, :]
y = y[ind_perm, :]
# X_train = X[ind_perm[0:(int(0.8 * X.shape[0]))], :, :, :]
# y_train = y[ind_perm[0:(int(0.8 * X.shape[0]))]]
#
# X_val = X[ind_perm[(int(0.8 * X.shape[0])) + 1:], :, :, :]
# y_val = y[ind_perm[(int(0.8 * X.shape[0])) + 1:]]
#

model = Sequential()

model.add(Conv2D(96, (5, 5), activation='relu', padding='same'))
model.add(MaxPooling2D((1, 5)))
model.add(BatchNormalization())
model.add(Dropout(0.25))

model.add(Conv2D(96, (5, 5), activation='relu', padding='same'))
model.add(MaxPooling2D((1, 2)))
model.add(BatchNormalization())
model.add(Dropout(0.25))

model.add(Conv2D(96, (5, 5), activation='relu', padding='same'))
model.add(MaxPooling2D((1, 2)))
model.add(BatchNormalization())
model.add(Dropout(0.25))

model.add(Conv2D(96, (5, 5), activation='relu', padding='same'))
model.add(MaxPooling2D((1, 2)))
model.add(BatchNormalization())
model.add(Dropout(0.25))

model.add(Reshape((865, 96)))
model.add(GRU(units=96, return_sequences=True, recurrent_dropout=0.25))
model.add(GRU(units=96, return_sequences=True, recurrent_dropout=0.25))

# Temporal max-pooling.
model.add(Reshape((865, 96, 1)))
model.add(MaxPooling2D((865, 1)))

model.add(Flatten())
model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer=Adam(learning_rate=0.001), loss=binary_crossentropy, metrics=[AUC()])

a = model(tf.random.normal([1, 865, 40, 1]))
model.summary()

checkpoint = ModelCheckpoint("best_model_865.hdf5", verbose=1, monitor='val_auc',
                             save_best_only=False, mode='auto', save_freq='epoch')

model.fit(X, y, batch_size=32, epochs=10, validation_split=0.2, callbacks=[checkpoint])