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
from data_helper import DataHelper

if __name__ == '__main__':
    train_data, test_data = DataHelper.load_preprocess_json()
    train_label = [doc for doc in train_data.cuisine]

    cos_dic = {}
    for label in train_label:
        if label in cos_dic:
            cos_dic[label] = cos_dic[label] + 1
        else:
            cos_dic[label] = 1

    for dic in cos_dic.items():
        print(dic)