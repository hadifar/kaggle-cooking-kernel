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
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.feature_selection.univariate_selection import SelectKBest, chi2
from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import LabelEncoder

from data_helper import DataHelper

if __name__ == '__main__':
    train_features, train_label, test_features = DataHelper.get_tfidf_vectorize()
    fselect = SelectKBest(chi2, k='all')
    train_features = fselect.fit_transform(train_features, train_label)
    test_features = fselect.transform(test_features)
    lb = LabelEncoder()
    train_label = lb.fit_transform(train_label)

    print('MNB')
    model1 = MultinomialNB(alpha=0.001)
    model1.fit(train_features, train_label)
    pred1 = model1.predict(test_features)
    pred1 = lb.inverse_transform(pred1)
    DataHelper.save_submission('feature_mnb', pred1)

    print('SGD')
    model2 = SGDClassifier(loss='modified_huber', n_iter=5, random_state=0, shuffle=True)
    model2.fit(train_features, train_label)
    pred2 = model2.predict(test_features)
    pred2 = lb.inverse_transform(pred2)
    DataHelper.save_submission('feature_sgd', pred2)

    print('Random forest')
    model3 = RandomForestClassifier()
    model3.fit(train_features, train_label)
    pred3 = model3.predict(test_features)
    pred3 = lb.inverse_transform(pred3)
    DataHelper.save_submission('feature_random_forest', pred3)

    print('GradientBoostingClassifier')
    model4 = GradientBoostingClassifier()
    model4.fit(train_features, train_label)
    pred4 = model4.predict(test_features)
    pred4 = lb.inverse_transform(pred4)
    DataHelper.save_submission('feature_gbc', pred4)
