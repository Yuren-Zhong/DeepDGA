
import numpy as np

from keras.models import Sequential
from keras.layers import Dense, Dropout, Embedding, LSTM, Bidirectional
from keras.layers import Activation, Conv1D, GlobalMaxPooling1D
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, LearningRateScheduler,TensorBoard

from models import basic_cnn_model
from data2json import load_data

maxlen = 100
batch_size = 256
epochs = 200

log_path = 'logs'
name = 'basic_cnn'

def build_callbacks(save_path, tflog_dir, batch_size):
    checkpoint = ModelCheckpoint(filepath=save_path, monitor="val_acc", verbose=1, save_best_only=True, save_weights_only=True)
    tf_log = TensorBoard(log_dir=tflog_dir, batch_size=batch_size)
    callbacks = [checkpoint, tf_log]
    return callbacks

print('Loading data...')
(x_train, y_train), (x_test, y_test) = load_data(200000, maxlen)
print(len(x_train), 'train sequences')
print(len(x_test), 'test sequences')

model = basic_cnn_model()

callbacks = build_callbacks(log_path+'/'+name+'_weights.h5', log_path+'/tf_log_'+name, batch_size)

model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

print('Train...')
model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          validation_data=[x_test, y_test],
          callbacks=callbacks)