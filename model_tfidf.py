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
import xgboost as xgb
from sklearn import linear_model
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import LinearSVC, SVC

from data_helper import DataHelper

if __name__ == '__main__':
    train_feature, train_label, test_feature = DataHelper.get_tfidf_vectorize()

    lb = LabelEncoder()
    train_label = lb.fit_transform(train_label)

    print('start logistic regression')
    logreg = LogisticRegression(C=10, solver='lbfgs', multi_class='multinomial', max_iter=1000, tol=1e-3)
    logreg.fit(train_feature, train_label)
    log_prediction = logreg.predict(test_feature)
    log_prediction = lb.inverse_transform(log_prediction)
    DataHelper.save_submission('logregression', log_prediction)

    print('start SGD')
    sgd = linear_model.SGDClassifier(random_state=0, max_iter=1000, tol=1e-3)
    sgd.fit(train_feature, train_label)
    sgd_prediction = sgd.predict(test_feature)
    sgd_prediction = lb.inverse_transform(sgd_prediction)
    DataHelper.save_submission('sgd', sgd_prediction)

    print('start Naive bayes')
    naive = MultinomialNB()
    naive.fit(train_feature, train_label)
    naive_prediction = naive.predict(test_feature)
    naive_prediction = lb.inverse_transform(naive_prediction)
    DataHelper.save_submission('naive_bayes', naive_prediction)

    print('start LinearSVC')
    linearsvm = LinearSVC(C=1.0, random_state=0, multi_class='crammer_singer', dual=False, max_iter=1500)
    linearsvm.fit(train_feature, train_label)
    linsvm_prediction = linearsvm.predict(test_feature)
    linsvm_prediction = lb.inverse_transform(linsvm_prediction)
    DataHelper.save_submission('linearsvm', linsvm_prediction)

    print("SVM 1vsRest")
    model = OneVsRestClassifier(SVC(C=100,
                                    gamma=1,
                                    coef0=1,
                                    decision_function_shape=None))

    model.fit(train_feature, train_label)
    svc_prediction = model.predict(test_feature)
    svc_prediction = lb.inverse_transform(svc_prediction)
    DataHelper.save_submission('svc', svc_prediction)

    print('XGBoost')
    xgboost = xgb.XGBClassifier(max_depth=6, n_estimators=1000, learning_rate=0.1
                                , min_child_weight=5,
                                gamma=1,
                                subsample=0.8,
                                colsample_bytree=0.8,
                                nthread=4,
                                scale_pos_weight=1,
                                )
    xgboost.fit(train_feature, train_label)
    xgb_prediction = xgboost.predict(test_feature)
    xgb_prediction = lb.inverse_transform(xgb_prediction)
    DataHelper.save_submission('xgb', xgb_prediction)
