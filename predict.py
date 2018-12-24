from keras.datasets import imdb
import numpy as np
from nltk import word_tokenize
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
from keras.preprocessing import sequence

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
set_session(tf.Session(config=config))

INDEX_FROM=3
word_to_id = imdb.get_word_index()
word_to_id = {k:(v+INDEX_FROM) for k,v in word_to_id.items()}
word_to_id["<PAD>"] = 0
word_to_id["<START>"] = 1
word_to_id["<UNK>"] = 2

model = tf.keras.models.load_model("./sentiment2.model.h5")

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
        test.append(word_to_id.get(word, word_to_id["<UNK>"]))
    return test

#predict("beautiful and touching movie rich colors great settings good acting and one of the most charming movies i have seen in a while i never saw such an interesting setting when i was in china my wife liked it so much she asked me to bfdsldsf on and rate it so other would enjoy too")
#predict("I hate this movie it's gross bad bad dont watch why ever did that")

