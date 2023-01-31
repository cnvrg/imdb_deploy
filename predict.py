# Copyright (c) 2023 Intel Corporation
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of
# this software and associated documentation files (the "Software"), to deal in
# the Software without restriction, including without limitation the rights to
# use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
# the Software, and to permit persons to whom the Software is furnished to do so,
# subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
# FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
# COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
# IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
# CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
#
# SPDX-License-Identifier: MIT

from keras.datasets import imdb
import numpy as np
import keras
from nltk import word_tokenize
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
from keras.preprocessing import sequence
import nltk
nltk.download('punkt')


config = tf.ConfigProto()
config.gpu_options.allow_growth = True
set_session(tf.Session(config=config))

INDEX_FROM=3
word_to_id = imdb.get_word_index()
word_to_id = {k:(v+INDEX_FROM) for k,v in word_to_id.items()}
word_to_id["<PAD>"] = 0
word_to_id["<START>"] = 1
word_to_id["<UNK>"] = 2

model = keras.models.load_model("sentiment2.model.h5")
model._make_predict_function()

reverse_word_index = dict([(value, key) for (key, value) in word_to_id.items()])

def decode_back_sentence(decoded):
    decoded_review = ' '.join([reverse_word_index[i] for i in decoded])
    return decoded_review

def predict(sentence):

    encoded = encode_sentence(sentence)

    pred = np.array([encoded])
    pred = vectorize_sequences(pred)
    a = model.predict(pred)
    return str(a[0][0])

def vectorize_sequences(sequences, dimension=10000):
    results = np.zeros((len(sequences), dimension))
    for i, sequence in enumerate(sequences):
        results[i, sequence] = 1.
    return results

def encode_sentence(sentence):
    test=[1]
    for word in word_tokenize(sentence):
        word_id = word_to_id.get(word, word_to_id["<UNK>"])
        if word_id > 9999:
            word_id = word_to_id["<UNK>"]
        test.append(word_id)
    return test
