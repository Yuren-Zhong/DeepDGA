
import numpy as np
import json

from models import basic_cnn_model
from dataset import remove_suffix, text2seq, pad_seq

maxlen = 100
batch_size = 256
epochs = 200

weights_path = 'logs/basic_cnn_weights.h5'

def process_line(line):
    return line.split('.')[0].replace(' ', '').replace('\n', '')

with open('data/other_dga.txt', 'r') as f:
  domains = f.readlines()

x = []
for d in domains:
    r = process_line(d)
    x.append(pad_seq(text2seq(r), maxlen))

x = np.array(x)

model = basic_cnn_model()
model.compile(loss='binary_crossentropy',
              optimizer='adam',
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