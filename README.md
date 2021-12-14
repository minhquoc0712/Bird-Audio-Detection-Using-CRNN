# Bird-Audio-Detection-Using-CRNN

**The file name and some folders need to be organized**.

Our model followed the proposed model in [1]. We had tried different settings for the model and the model provides the final result is presented.


Pre-processed data: Training data is loaded using a sampling rate of 44100 Hz. Then the array is normalized to the range [-1, 1]. Secondly, for each audio file, we take the 40 bins, log mel-band energy feature with Hamming window, each window consists of 2048 samples and use 50% overlap between windows, corresponding to approximately 46 ms window length, close to the implementation in [1]. Since the varying size of each audio, we chose the number of frames in each data example as 450. When the number of frames is larger than 450, we remove that example. There are about 50 examples to be removed from the training set. And when the number of frames is smaller than 450, we pad the spectrogram with the value -15 for all frequency bins. The value -15 is chosen from the observation that it will be the minimum value in the spectrograms of all training examples. The number of frames 450 is chosen because, on one hand, we want the number to exclude as little as possible the oversize example, and on the other hand, we don't want to pad too much on the spectrogram. The test data are audio files with sampling rate of 48 kHz. To make the spectrograms have the same size as the training data (450, 40), the test data is processed as the training data but now using the sampling rate of 48 kHz and hop length of 1067 samples. 


Network architecture: CRNN in [1] obtains the second place in BAD challenge. CRNN is also observed to give a considerable improvement on different datasets consisting of everyday sound events [2]. So we choose CRNN as our model. The model consists of four stages:


CNN layers are made from four similar blocks. Each block is a sequence of a convolutional layer having kernel size of (5, 5), 96 output channel, with ReLU activations, a non-overlapping max-pooling over the frequency axis, a batch normalization layer, and a drop-out layer with the rate of 0.25. The max-pooling size of the four blocks is 5, 2, 2, and 2.
Two GRU layers with input size (450, 96), and the drop-out rate is 0.25.
A temporal max-pooling layer.
A single neural, using sigmoid activation. 


Since CNN performs very well in recognizing the patterns, we used it in the first stage on the whole Mel spectrogram to extract useful features. The GRU layers are good at learning the longer term temporal context in the audio signals. Stage 3-4 are used to create the classification output as the probability of the presence of birds sound in an example. The network is trained with Adam optimizer with the learning rate of 0.0005 and binary cross-entropy as the loss function. Training data is divided by a ratio of 80% used for training and 20% used for validation.


Result and comment: To obtain the best parameters, we try different size of input spectrogram: 862 frames and 40 mel-frequency bins, learning rate: 0.001, 0.0001, and 0.0008, the dropout rate in GRU layers: 0.5. By unknown problem, the value of loss function and AUC score on both training and validation set are improved after many epochs, but the AUC score from the submission decreases right after the second epoch. So our model is trained from 1 to 10 epochs and we using the best final result. The final score is around 0.66-0.67 on the public leaderboard. After the deadline, we realize that the problem of not improving AUC score on the test set may come from the difference in hop-length used for training data and testing data. But this is not verified yet.

References: 

[1]  E. Çakır, S. Adavanne, G. Parascandolo, K. Drossos, and T. Virtanen, "Convolutional recurrent neural networks for bird audio detection," in EUSIPCO 2020.

[2]  E. Cakir, G. Parascandolo, T. Heittola, H. Huttunen, and T. Virtanen, “Convolutional recurrent neural networks for polyphonic sound event detection,” in IEEE/ACM TASLP Special Issue on Sound Scene and Event Analysis, 2017, accepted for publication.
