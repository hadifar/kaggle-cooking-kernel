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
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import LabelEncoder
from tensorflow.python.keras import Sequential
from tensorflow.python.keras.layers import Dense, Dropout
import numpy as np

from data_helper import DataHelper

if __name__ == '__main__':
    train_feature, train_label, test_feature = DataHelper.get_tfidf_vectorize()

    lb = LabelEncoder()
    train_label = lb.fit_transform(train_label)

    svd = TruncatedSVD(100)
    train_words_vectors = svd.fit_transform(train_feature)
    test_words_vectors = svd.fit_transform(test_feature)

    model = Sequential()
    model.add(Dense(256, input_dim=100, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(20, activation="softmax"))
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    history = model.fit(train_words_vectors, train_label, epochs=50, batch_size=256)

    print("Predict on test data ... ")
    prediction = model.predict(test_feature)
    prediction = np.argmax(prediction, axis=1)
    mlp_prediction = lb.inverse_transform(prediction)
    DataHelper.save_submission('mlp_svd', mlp_prediction)
