import numpy as np
import pandas as pd
import emoji

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

import tensorflow
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Dense, Input, Dropout, LSTM, Activation
from keras.layers.embeddings import Embedding
from tensorflow.keras.callbacks import ReduceLROnPlateau

from setup import read_glove_vecs
from setup import read_data
from setup import sentences_to_indices
from setup import pretrained_embedding_layer

from additional import Emojify

word_to_index, index_to_word, word_to_vec_map = read_glove_vecs('data/glove.6B.100d.txt')

X_train, Y_train = read_data('data/train_emoji.csv')

maxWords = len(max(X_train, key=len).split())

X_train_indices = sentences_to_indices(X_train, word_to_index, maxWords)

emojifier = Emojify((maxWords,), word_to_vec_map, word_to_index)

reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.9, patience=3, min_lr=0.00001, verbose=1)
emojifier.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
emojifier.fit(X_train_indices, Y_train, epochs = 100, batch_size = 16, shuffle=True, callbacks=[reduce_lr])

emojifier.save_weights('saved_model/')
