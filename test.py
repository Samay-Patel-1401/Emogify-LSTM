import numpy as np
import pandas as pd
import emoji

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

import tensorflow
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input, Dropout, LSTM, Activation
from keras.layers.embeddings import Embedding
from tensorflow.keras.callbacks import ReduceLROnPlateau

from setup import read_glove_vecs
from setup import read_data
from setup import sentences_to_indices
from setup import pretrained_embedding_layer

from additional import Emojify
from additional import label_to_emoji

word_to_index, index_to_word, word_to_vec_map = read_glove_vecs('data/glove.6B.100d.txt')

X_test, Y_test = read_data('data/test_emoji.csv')

maxWords = len(max(X_test, key=len).split())

X_test_indices = sentences_to_indices(X_test, word_to_index, maxWords)

test = Emojify((maxWords,), word_to_vec_map, word_to_index)
test.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

test.load_weights('saved_model/')

loss, acc = test.evaluate(X_test_indices, Y_test, verbose=2)
print()
print("Test accuracy = ", acc)

# x_test = np.array(['not feeling happy','no one knows America better than Trump', 'I want to have lunch with you', 'I love playing basketball', 'I love China',
#                    'I love yangping'])
# X_test_indices = sentences_to_indices(x_test, word_to_index, maxWords)
# pred = test.predict(X_test_indices)
# for i in range(len(x_test)):
#     num = np.argmax(pred[i])
#     print(x_test[i] + ' ' + label_to_emoji(num).strip())
