## imports

import tensorflow as tf
import theano
import pandas as pd
import numpy as np
from keras.layers import  LSTM, Embedding, Activation, Lambda, Bidirectional
from keras.engine import Input, Model, InputSpec
from keras.preprocessing.sequence import pad_sequences
from keras.optimizers import Adam
from keras import regularizers
from sklearn.utils import class_weight
from keras.callbacks import EarlyStopping
from keras import backend as K
from keras.preprocessing import sequence
from keras.models import model_from_json
import os
from gensim import models
from keras.callbacks import ModelCheckpoint
from keras.layers import Dense, Dropout, Reshape, Flatten, concatenate, Input, Conv1D, GlobalMaxPooling1D, Embedding
from keras.layers.recurrent import LSTM
from keras.models import Sequential
from keras.preprocessing.text import Tokenizer
from keras.models import Model
from sklearn.model_selection import train_test_split

INPUT_DIM = 4 # a vocabulary of 4 words in case of fnn sequence (ATCG)
OUTPUT_DIM = 50 # Embedding output
MAXLEN = 300 # cuts text after number of these characters in pad_sequences
max_sequence_length = 300


input_file = '/content/b11 (2).csv'
def letter_to_index(letter):
    _alphabet = 'ATGC'
    return next((i for i, _letter in enumerate(_alphabet) if _letter == letter), 0)

def load_data(test_split = 0.2, maxlen = MAXLEN):
    print ('Loading data...')
    df = pd.read_csv(input_file)
    df['sequence'] = df['sequence'].apply(lambda x: [int(letter_to_index(e)) for e in x])
    df = df.reindex(np.random.permutation(df.index))
    train_size = int(len(df) * (1 - test_split))
    X_train = df['sequence'].values[:train_size]
    y_train = np.array(df['target'].values[:train_size])
    X_test = np.array(df['sequence'].values[train_size:])
    y_test = np.array(df['target'].values[train_size:])
    return pad_sequences(X_train, maxlen=maxlen), y_train, pad_sequences(X_test, maxlen=maxlen), y_test


X_train, y_train, X_test, y_test = load_data()    


def model(input_length, output_dim = OUTPUT_DIM, input_dim = INPUT_DIM):
  embedding_layer = Embedding(input_dim = INPUT_DIM, output_dim = output_dim, input_length = input_length, name='embedding_layer')
  sequence_input = Input(shape=(max_sequence_length,), dtype='int32')
  embedded_sequences = embedding_layer(sequence_input)
  convs = []
  filter_sizes = [2,3,4]
  for filter_size in filter_sizes:
    l_conv = Conv1D(filters=200, kernel_size=filter_size, activation='relu')(embedded_sequences)
    lstm_layer = Bidirectional(LSTM(32, return_sequences=True))(l_conv)
    l_pool = GlobalMaxPooling1D()(lstm_layer)
    convs.append(l_pool)

  l_merge = concatenate(convs, axis=1)
  x = Dropout(0.1)(l_merge)  
  x = Dense(128, activation='relu')(x)
  preds = Dense(1, activation='sigmoid')(x)

  model = Model(sequence_input, preds)
  model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['acc'])
  model.summary()
  return model


model = model(len(X_train[0])) 

history = model.fit(X_train, y_train, batch_size=128, epochs=50, validation_split = 0.1, verbose = 1 )
