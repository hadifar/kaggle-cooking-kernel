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
import re

import pandas as pd
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer


class DataHelper:
    dataset_path = '/Users/mac/PycharmProjects/cooking-kernel/dataset/'
    word_index_name = 'word_to_index'

    @staticmethod
    def load_data():
        train_df = pd.read_json(DataHelper.dataset_path + 'train.json')
        test_df = pd.read_json(DataHelper.dataset_path + 'test.json')

        train_df['ingredients'] = train_df.ingredients.apply(lambda x: [str.lower(item) for item in x])
        test_df['ingredients'] = test_df.ingredients.apply(lambda x: [str.lower(item) for item in x])

        train_df['ingredients'] = train_df.ingredients.apply(
            lambda x: [re.sub(r'[^\x00-\x7F]+', ' ', item) for item in x])

        test_df['ingredients'] = test_df.ingredients.apply(
            lambda x: [re.sub(r'[^\x00-\x7F]+', ' ', item) for item in x])

        train_df['ingredients'] = train_df.ingredients.apply(
            lambda x: [" ".join(item.replace('-', ' ').split(' ')) for item in x])

        test_df['ingredients'] = test_df.ingredients.apply(
            lambda x: [" ".join(item.replace('-', ' ').split(' ')) for item in x])

        lemmatizer = WordNetLemmatizer()

        train_df['ingredients'] = train_df.ingredients.apply(
            lambda x: [lemmatizer.lemmatize(token) for item in x for token in item.split()])

        test_df['ingredients'] = test_df.ingredients.apply(
            lambda x: [lemmatizer.lemmatize(token) for item in x for token in item.split()])

        return train_df, test_df

    @staticmethod
    def load_preprocess_json():
        train_df = DataHelper._load_train()
        test_df = DataHelper._load_test()
        return train_df, test_df

    @staticmethod
    def _load_test():
        return pd.read_json(DataHelper.dataset_path + 'test2.json')

    @staticmethod
    def _load_train():
        return pd.read_json(DataHelper.dataset_path + 'train2.json')

    @staticmethod
    def generate_text(data):
        text_data = [" ".join(doc) for doc in data.ingredients]
        return text_data

    @staticmethod
    def get_tfidf_vectorize():
        print('start vectorized...')
        train_df, test_df = DataHelper.load_preprocess_json()
        vect = TfidfVectorizer()

        train_features = vect.fit_transform(DataHelper.generate_text(train_df))
        test_features = vect.transform(DataHelper.generate_text(test_df))

        train_label = [doc for doc in train_df.cuisine]
        print('finish vectorized...')

        return train_features, train_label, test_features

    @staticmethod
    def get_bigram_vectorize():
        print('start vectorized...')
        train_df, test_df = DataHelper.load_preprocess_json()
        vect = CountVectorizer(ngram_range=(1, 2))

        train_features = vect.fit_transform(DataHelper.generate_text(train_df))
        test_features = vect.transform(DataHelper.generate_text(test_df))

        train_label = [doc for doc in train_df.cuisine]
        print('finish vectorized...')

        return train_features, train_label, test_features

    @staticmethod
    def get_unigram_vectorize():
        print('start vectorized...')
        train_df, test_df = DataHelper.load_preprocess_json()
        vect = CountVectorizer()

        train_features = vect.fit_transform(DataHelper.generate_text(train_df))
        test_features = vect.transform(DataHelper.generate_text(test_df))

        train_label = [doc for doc in train_df.cuisine]
        print('finish vectorized...')

        return train_features, train_label, test_features

    @staticmethod
    def save_submission(file_name, y_pred):
        test_df = DataHelper._load_test()
        # Submission
        print("Generate Submission File for ", file_name)
        test_id = [doc for doc in test_df.id]
        sub = pd.DataFrame({'id': test_id, 'cuisine': y_pred}, columns=['id', 'cuisine'])
        sub.to_csv(file_name + '_output.csv', index=False)
