
import numpy as np
import json

from keras.optimizers import Adam

from models import basic_cnn_model, lstm_model
from dataset import remove_suffix, text2seq, pad_seq

maxlen = 100
batch_size = 256
epochs = 200

def process_data(domains, filter=True):
    x = []
    if filter == False:
        for d in domains:
            x.append(pad_seq(text2seq(d), maxlen))
        return np.array(x)

    for d in domains:
        r = remove_suffix(d)
        if r is not None:
            x.append(pad_seq(text2seq(d), maxlen))
        else:
            # can't guarantee accuracy
            x.append(pad_seq(text2seq(d.split('.')[0]), maxlen))

    return np.array(x)

weights_path = 'logs/basic_lstm_weights.h5'

domains = json.load(open('domains.json'))

x = process_data(domains, filter=False)

model = lstm_model()
model.compile(loss='binary_crossentropy',
              optimizer=Adam(lr=0.01),
              metrics=['accuracy'])
model.load_weights(weights_path, by_name=True)

result = model.predict(x,
                   batch_size=batch_size,
                   verbose=0) 

print("*"*40)
print("weights path : " + weights_path)
print("*"*40)
i = 0
for d in domains:
    if result[i][0] < 0.5:
        r = 'legit'
    else:
        r = 'dga'
    print(d + ' : ' + r +  ' : ' + str(result[i][0]))
    i += 1