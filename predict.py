
import numpy as np
import json

from keras.models import Sequential
from keras.layers import Dense, Dropout, Embedding, LSTM, Bidirectional
from keras.layers import Activation, Conv1D, GlobalMaxPooling1D
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, LearningRateScheduler,TensorBoard

from models import basic_cnn_model
from data2json import load_data, text2seq, pad_seq

maxlen = 100
batch_size = 256
epochs = 200

weights_path = 'logs/basic_cnn_weights.h5'


domains = json.load(open('domains.json'))
x = []
for d in domains:
	x.append(pad_seq(text2seq(d), maxlen))

x = np.array(x)

model = basic_cnn_model()
model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
model.load_weights(weights_path, by_name=True)

result = model.predict(x,
                   batch_size=batch_size,
                   verbose=0) 

i = 0
for d in domains:
	if result[i][0] < 0.5:
		r = 'legit'
	else:
		r = 'dga'
	print(d + " : " + r)
	i += 1