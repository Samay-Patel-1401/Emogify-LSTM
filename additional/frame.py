import emoji
import numpy as np
import pandas as pd

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

LABELS = {
    0: ':heart:',
    1: ':baseball:',
    2: ':smile:',
    3: ':disappointed:',
    4: ':fork_and_knife:'
}

import tensorflow
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input, Dropout, LSTM, Activation
from keras.layers.embeddings import Embedding

from setup import pretrained_embedding_layer

def label_to_emoji(label):
    return emoji.emojize(LABELS[str(label)], use_aliases=True)

def Emojify(input_shape, word_to_vec_map, word_to_index):
    sentence_indices = Input(shape=input_shape, dtype='int32')
    embedding_layer = pretrained_embedding_layer(word_to_vec_map, word_to_index)
    embeddings = embedding_layer(sentence_indices)

    X = LSTM(128, return_sequences=True)(embeddings)
    X = Dropout(0.8)(X)
    X = LSTM(128, return_sequences=False)(X)
    X = Dropout(0.8)(X)
    X = Dense(5, activation='softmax')(X)
    X = Activation('softmax')(X)

    model = Model(inputs=sentence_indices, outputs=X)

    return model
