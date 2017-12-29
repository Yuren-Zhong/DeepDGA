
import numpy as np

from keras.optimizers import Adam
from keras.models import Sequential
from keras.layers import Dense, Dropout, Embedding, LSTM, Bidirectional
from keras.layers import Activation, Conv1D, GlobalMaxPooling1D
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, LearningRateScheduler,TensorBoard

from models import basic_cnn_model, lstm_model, cnn_lstm_model, bidirectional_lstm_model
from dataset import load_data

maxlen = 100
batch_size = 256
epochs = 200

log_path = 'logs'
name = 'basic_bidirectional_lstm'

def lr_schedule(epoch):
    """Learning Rate Schedule
    Learning rate is scheduled to be reduced after 80, 120, 160, 180 epochs.
    Called automatically every epoch as part of callbacks during training.
    # Arguments
        epoch (int): The number of epochs
    # Returns
        lr (float32): learning rate
    """
    lr = 1e-3
    if epoch > 180:
        lr *= 0.5e-3
    elif epoch > 160:
        lr *= 1e-3
    elif epoch > 120:
        lr *= 1e-2
    elif epoch > 80:
        lr *= 1e-1
    print('Learning rate: ', lr)
    return lr

def build_callbacks(save_path, tflog_dir, batch_size):
    checkpoint = ModelCheckpoint(filepath=save_path, monitor="val_acc", verbose=1, save_best_only=True, save_weights_only=True)
    lr_reducer = ReduceLROnPlateau(factor=0.1,
                                cooldown=0,
                                patience=5,
                                min_lr=0.5e-6)
    lr_scheduler = LearningRateScheduler(lr_schedule)
    # tf_log = TensorBoard(log_dir=tflog_dir, batch_size=batch_size)
    callbacks = [checkpoint, lr_scheduler]
    return callbacks

print('Loading data...')
(x_train, y_train), (x_test, y_test) = load_data(200000, maxlen, filter=True)
print(len(x_train), 'train sequences')
print(len(x_test), 'test sequences')

model = bidirectional_lstm_model()
model.summary()

callbacks = build_callbacks(log_path+'/'+name+'_weights.h5', log_path+'/tf_log_'+name, batch_size)
'''
model.compile(loss='binary_crossentropy',
              optimizer=Adam(lr=0.001),
              metrics=['accuracy'])
'''
model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

print('Train...')
model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          validation_data=[x_test, y_test],
          callbacks=callbacks)