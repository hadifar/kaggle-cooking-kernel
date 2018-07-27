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
import pickle
import re

import pandas as pd
from nltk.stem import WordNetLemmatizer


class DataHelper:
    dataset_path = '/Users/mac/PycharmProjects/cooking-kernel/dataset/'
    word_index_name = 'word_to_index'

    def __init__(self):
        self.convert_json_to_df()

    @staticmethod
    def convert_json_to_df():
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

        train_df.to_json(DataHelper.dataset_path + 'train2.json', orient='records')
        test_df.to_json(DataHelper.dataset_path + 'test2.json', orient='records')

        return train_df, test_df

    @staticmethod
    def load_preprocess_json():
        train_df = pd.read_json(DataHelper.dataset_path + 'train2.json')
        test_df = pd.read_json(DataHelper.dataset_path + 'test2.json')
        return train_df, test_df

    @staticmethod
    def save_index(word_index_dic):
        with open(DataHelper.word_index_name + '.pkl', 'wb') as f:
            pickle.dump(word_index_dic, f, pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def load_index_dictionary():
        with open(DataHelper.word_index_name + '.pkl', 'rb') as f:
            return pickle.load(f)
