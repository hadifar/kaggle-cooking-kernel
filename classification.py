# -*- coding: utf-8 -*-
#
# Copyright 2018 Amir Hadifar. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from tensorflow import keras

from data_helper import DataHelper


def generate_text(data):
    text_data = [" ".join(doc) for doc in data.ingredients]
    return text_data


if __name__ == '__main__':
    print('load data..')
    train_data, test_data = DataHelper.load_preprocess_json()
    target = [doc for doc in train_data.cuisine]
    lb = LabelEncoder()
    train_y = lb.fit_transform(target)

    train_text = generate_text(train_data)
    test_text = generate_text(test_data)
    tokenizer = keras.preprocessing.text.Tokenizer()
    tokenizer.fit_on_texts(train_text + test_text)
    print(len(tokenizer.word_index))

    train_seq = tokenizer.texts_to_sequences(train_text)
    test_seq = tokenizer.texts_to_sequences(test_text)

    train_seq = keras.preprocessing.sequence.pad_sequences(train_seq, padding='post', maxlen=145)
    test_seq = keras.preprocessing.sequence.pad_sequences(test_seq, padding='post', maxlen=145)
    print(test_seq.shape)

    # input shape is the vocabulary count used for the movie reviews (10,000 words)
    vocab_size = len(tokenizer.word_index) + 1

    model = keras.Sequential()

    model.add(keras.layers.Embedding(vocab_size, 64))
    model.add(keras.layers.GlobalAveragePooling1D())
    model.add(keras.layers.Dense(200, activation=tf.nn.relu, input_dim=145))
    model.add(keras.layers.Dense(100, activation=tf.nn.relu))
    model.add(keras.layers.Dense(20, activation=tf.nn.softmax))

    print(model.summary())

    model.compile(optimizer=tf.train.AdamOptimizer(),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    print("train model...")
    history = model.fit(train_seq,
                        train_y,
                        epochs=50,
                        batch_size=256,
                        validation_split=0.01,
                        verbose=1)

    print("Predict on test data ... ")
    prediction = model.predict(test_seq)
    prediction = np.argmax(prediction, axis=1)
    y_pred = lb.inverse_transform(prediction)

    # Submission
    print("Generate Submission File ... ")
    test_id = [doc for doc in test_data.id]
    sub = pd.DataFrame({'id': test_id, 'cuisine': y_pred}, columns=['id', 'cuisine'])
    sub.to_csv('svm_output.csv', index=False)
