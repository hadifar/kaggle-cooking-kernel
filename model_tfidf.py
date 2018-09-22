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
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from data_helper import DataHelper

if __name__ == '__main__':
    # train_feature, train_label, test_feature = DataHelper.get_bigram_vectorize()
    train_feature, train_label, test_feature = DataHelper.get_tfidf_vectorize()

    lb = LabelEncoder()
    train_label = lb.fit_transform(train_label)

    print("SVM 1vsRest")
    model = OneVsRestClassifier(SVC(C=100,
                                    gamma=1,
                                    coef0=1,
                                    decision_function_shape=None))

    model.fit(train_feature, train_label)
    svc_prediction = model.predict(test_feature)
    svc_prediction = lb.inverse_transform(svc_prediction)
    DataHelper.save_submission('svc', svc_prediction)