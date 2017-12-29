
from keras.models import Sequential
from keras.layers import Dense, Dropout, Embedding, LSTM, CuDNNLSTM, Bidirectional
from keras.layers import Activation, Conv1D, GlobalMaxPooling1D, MaxPooling1D

max_features = 5000
embedding_dims = 128
maxlen = 100

def basic_cnn_model():
    global max_features
    global embedding_dims
    global maxlen

    filters = 250
    kernel_size = 3
    hidden_dims = 250
    batch_size = 256

    model = Sequential()
    model.add(Embedding(max_features, embedding_dims, input_length=maxlen))
    model.add(Dropout(0.2))
    model.add(Conv1D(filters,
                 kernel_size,
                 padding='valid',
                 activation='relu',
                 strides=1))
    model.add(GlobalMaxPooling1D())
    model.add(Dense(hidden_dims))
    model.add(Dropout(0.2))
    model.add(Activation('relu'))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))

    return model

def lstm_model():
    global max_features
    global embedding_dims
    global maxlen

    model = Sequential()
    model.add(Embedding(max_features, embedding_dims, input_length=maxlen))
    model.add(LSTM(128))
    model.add(Dropout(0.5))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))

    return model

def bidirectional_lstm_model():
    global max_features
    global embedding_dims
    global maxlen

    model = Sequential()
    model.add(Embedding(max_features, embedding_dims, input_length=maxlen))
    model.add(Bidirectional(LSTM(128)))
    model.add(Dropout(0.5))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))

    return model

def cnn_lstm_model():
    global max_features
    global embedding_dims
    global maxlen

    kernel_size = 5
    filters = 64
    pool_size = 4

    model = Sequential()
    model.add(Embedding(max_features, embedding_dims, input_length=maxlen))
    model.add(Dropout(0.25))
    model.add(Conv1D(filters,
                     kernel_size,
                     padding='valid',
                     activation='relu',
                     strides=1))
    model.add(MaxPooling1D(pool_size=pool_size))
    model.add(LSTM(128))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))

    return model