
import numpy as np
import json

from keras.optimizers import Adam

from models import basic_cnn_model
from dataset import remove_suffix, text2seq, pad_seq

maxlen = 100
batch_size = 256
epochs = 200

weights_path = 'logs/basic_cnn_weights.h5'
dga_file = 'data/360netlab_dga.txt'

print("dga file : " + dga_file)
print("weights path : " + weights_path)

def process_line(line, filter=True):
    if filter:
        return line.split('.')[0].replace(' ', '').replace('\n', '')
    return line.replace(' ', '').replace('\n', '')

with open(dga_file, 'r') as f:
  domains = f.readlines()

x = []
for d in domains:
    r = process_line(d, filter=True)
    x.append(pad_seq(text2seq(r), maxlen))

x = np.array(x)

model = basic_cnn_model()
model.compile(loss='binary_crossentropy',
              optimizer=Adam(lr=0.01),
              metrics=['accuracy'])
model.load_weights(weights_path, by_name=True)

result = model.predict(x,
                   batch_size=batch_size,
                   verbose=0) 

print("*"*40)
n = 0
for i in range(len(x)):
    if result[i][0] > 0.5:
        n += 1
    i += 1

print(n, len(x))
print(n/len(x))