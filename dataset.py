import numpy as np

from keras.preprocessing.text import Tokenizer
from random import shuffle
import json

table = {
    '0': 0,
    '1': 1,
    '2': 2,
    '3': 3,
    '4': 4,
    '5': 5,
    '6': 6,
    '7': 7,
    '8': 8,
    '9': 9,
    '.': 10,
    'a': 11,
    'b': 12,
    'c': 13,
    'd': 14,
    'e': 15,
    'f': 16,
    'g': 17,
    'h': 18,
    'i': 19,
    'j': 20,
    'k': 21,
    'l': 22,
    'm': 23,
    'n': 24,
    'o': 25,
    'p': 26,
    'q': 27,
    'r': 28,
    's': 29,
    't': 30,
    'u': 31,
    'v': 32,
    'w': 33,
    'x': 34,
    'y': 35,
    'z': 36,
    '-': 37,
    '_': 38
}

suffixes = ['.com', '.net', '.biz', '.ru', '.org', '.co.uk', '.info', '.cc', '.ws', '.cn']

pad_value = 40

def text2seq(text):
    s = []
    for c in text:
        s.append(table[c])
    return s

def pad_seq(s, max_len):
    global pad_value
    if len(s) > max_len:
        return s[:max_len]
    for i in range(len(s), max_len):
        s.append(pad_value)
    return s

def suffix_in(domain):
    global suffixes
    for s in suffixes:
        if s in domain:
            return s
    return None

def remove_suffix(domain):
    ret = suffix_in(domain)
    if ret is not None:
        return domain.replace(ret, '')
    return None

def load_data(val_number, max_len, filter=True):
    global suffixes

    with open('data/all_dga.txt', 'r') as fneg:
        neg_raw_data = fneg.readlines()
    with open('data/all_legit.txt', 'r') as fpos:
        pos_raw_data = fpos.readlines()

    x_data = []
    all_data = {}

    if filter == True:
        for line in neg_raw_data:
            x = line.split(' ')[0]
            r = remove_suffix(x)
            if r is not None:
                x_data.append(r)
                all_data[r] = 1
        for line in pos_raw_data:
            x = line.split(' ')[0]
            r = remove_suffix(x)
            if r is not None:
                x_data.append(r)
                all_data[r] = 0
    else:
        for line in neg_raw_data:
            x = line.split(' ')[0]
            x_data.append(x)
            all_data[x] = 1
        for line in pos_raw_data:
            x = line.split(' ')[0]
            x_data.append(x)
            all_data[x] = 0


    shuffle(x_data)

    x_train = []
    x_test = []
    y_train = []
    y_test = []

    n = 0
    for x in x_data:
        m = pad_seq(text2seq(x), max_len)
        if n < val_number:
            x_test.append(m)
            y_test.append(all_data[x])
        else:
            x_train.append(m)
            y_train.append(all_data[x])
        n += 1

    return (x_train, y_train), (x_test, y_test)

def test():
    with open("x_train.json", 'r') as f:
        d1 = json.load(f)
    with open("x_test.json", 'r') as f:
        d2 = json.load(f)
    with open("y_train.json", 'r') as f:
        d3 = json.load(f)
    with open("y_test.json", 'r') as f:
        d4 = json.load(f)

    print(len(d1), len(d2), len(d3), len(d4))

    l1 = len(d1[0])
    for d in d1:
        if len(d) == l1:
            continue
        else:
            l1 = (l1, len(d))
            break
    l2 = len(d2[0])
    for d in d2:
        if len(d) == l2:
            continue
        else:
            l2 = (l2, len(d))
            break

    print(l1, l2)

if __name__ == '__main__':
    print("dataset main")