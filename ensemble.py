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
import os
import warnings

import numpy as np
import xgboost as xgb
from sklearn.cross_validation import KFold
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import classification_report
from sklearn.multiclass import OneVsRestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC

from data_helper import DataHelper

warnings.filterwarnings("ignore", category=DeprecationWarning)

train_feature, train_label, test_feature = DataHelper.get_tfidf_vectorize()
lda_train, lda_test = DataHelper.get_lda_topics(train_feature, test_feature)


# train_feature, train_label, test_feature = DataHelper.get_bigram_vectorize()

# svd = TruncatedSVD(n_components=2, n_iter=10)
#
# train_svd_feature, test_svd_feature = svd.fit_transform(train_feature), svd.fit_transform(test_feature)
#
# train_feature, test_feature = train_svd_feature, test_svd_feature


class SklearnHelper(object):
    def __init__(self, clf, seed=0, params=None):
        if seed != -1:
            params['random_state'] = seed
        self.clf = clf(**params)

    def train(self, x_train, y_train):
        self.clf.fit(x_train, y_train)

    def predict(self, x):
        return self.clf.predict(x)

    def fit(self, x, y):
        return self.clf.fit(x, y)

    def feature_selection(self, x_train, x_test):
        model = SelectFromModel(self.clf, prefit=True)
        x_train_ = model.transform(x_train)
        x_test_ = model.transform(x_test)
        return x_train_, x_test_


def get_oof(clf, x_train, y_train, x_test):
    print('get_oof started...')
    oof_train = np.zeros((ntrain,))
    oof_test = np.zeros((ntest,))
    oof_test_skf = np.empty((NFOLDS, ntest))

    for i, (train_index, test_index) in enumerate(kf):
        x_tr = x_train[train_index]
        y_tr = y_train[train_index]
        x_te = x_train[test_index]

        clf.train(x_tr, y_tr)

        oof_train[test_index] = clf.predict(x_te)
        oof_test_skf[i, :] = clf.predict(x_test)

    oof_test[:] = oof_test_skf.mean(axis=0)
    return oof_train.reshape(-1, 1), oof_test.reshape(-1, 1)


NFOLDS = 3  # set folds for out-of-fold prediction
SEED = 0
lb = LabelEncoder()
train_label = lb.fit_transform(train_label)

# todo: fix this
X_train = train_feature[:35000]
Y_train = train_label[:35000]
X_test = train_feature[35000:]
Y_test = train_label[35000:]

ntrain = X_train.shape[0]
ntest = X_test.shape[0]
kf = KFold(ntrain, n_folds=NFOLDS, random_state=SEED)

del train_feature
del test_feature
del train_label

rf_params = {
    'n_jobs': -1,
    'n_estimators': 1000,
    'max_depth': 30,
    'min_samples_leaf': 2,
    'max_features': 'sqrt'
}

# Extra Trees Parameters
et_params = {
    'n_jobs': -1,
    'n_estimators': 1000,
    'max_depth': 128,
    'min_samples_leaf': 2,
    'max_features': 'sqrt'
}

# Gradient Boosting parameters
gb_params = {
    'n_estimators': 1000,
    'max_depth': 30,
    'min_samples_leaf': 2,
}

# Support Vector Classifier parameters
svc_params = {
    'kernel': 'linear',
    'C': 0.025
}

xgb_params = {
    'n_estimators': 2000,
    'max_depth': 4,
    'min_child_weight': 2,
    'gamma': 0.9,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'objective': 'binary:logistic',
    'nthread': -1,
    'scale_pos_weight': 1
}

ova_params = {
    'estimator': SVC(C=50, gamma=1.4, coef0=1),
    'n_jobs': -1
}

mnb_params = {
    'alpha': 1.0
}

sgd_params = {
    'loss': 'log'
}

knn_params = {
    'weights': 'distance',
    # todo: 60
    'n_neighbors': 2
}

rf = SklearnHelper(clf=RandomForestClassifier, seed=SEED, params=rf_params)
et = SklearnHelper(clf=ExtraTreesClassifier, seed=SEED, params=et_params)
gb = SklearnHelper(clf=GradientBoostingClassifier, seed=SEED, params=gb_params)
svc = SklearnHelper(clf=SVC, seed=SEED, params=svc_params)
ova_svm = SklearnHelper(clf=OneVsRestClassifier, seed=-1, params=ova_params)
mnb = SklearnHelper(MultinomialNB, seed=-1, params=mnb_params)
sgd = SklearnHelper(SGDClassifier, seed=-1, params=sgd_params)
knn = SklearnHelper(KNeighborsClassifier, seed=-1, params=knn_params)

# Create our OOF train and test predictions. These base results will be used as new features
et_oof_train, et_oof_test = get_oof(et, X_train, Y_train, X_test)  # Extra Trees
rf_oof_train, rf_oof_test = get_oof(rf, X_train, Y_train, X_test)  # Random Forest
gb_oof_train, gb_oof_test = get_oof(gb, X_train, Y_train, X_test)  # Gradient Boost
svc_oof_train, svc_oof_test = get_oof(svc, X_train, Y_train, X_test)  # Support Vector Classifier

ova_svm_oof_train, ova_svm_oof_test = get_oof(ova_svm, X_train, Y_train, X_test)
mnb_oof_train, mnb_oof_test = get_oof(mnb, X_train, Y_train, X_test)
sgd_oof_train, sgd_oof_test = get_oof(sgd, X_train, Y_train, X_test)
knn_oof_train, knn_oof_test = get_oof(knn, X_train, Y_train, X_test)

if os.path.isfile('./dataset/new_x_train.npy'):
    print('load cache... feature selection...')
    new_x_train = np.load('./dataset/new_x_train.npy')
    new_x_test = np.load('./dataset/new_x_test.npy')
else:
    print('feature selection started...')
    rf_f_train, rf_f_test = rf.feature_selection(X_train, X_test)
    et_f_train, et_f_test = et.feature_selection(X_train, X_test)
    gb_f_train, gb_f_test = gb.feature_selection(X_train, X_test)
    svc_f_train, svc_f_test = svc.feature_selection(X_train, X_test)
    ova_svm_f_train, ova_svm_f_test = ova_svm.feature_selection(X_train, X_test)
    mnb_f_train, mnb_f_test = mnb.feature_selection(X_train, X_test)
    sgd_f_train, sgd_f_test = sgd.feature_selection(X_train, X_test)
    knn_f_train, knn_f_test = knn.feature_selection(X_train, X_test)

    new_x_train = np.concatenate([rf_f_train, et_f_train,
                                  gb_f_train, svc_f_train,
                                  ova_svm_f_train, mnb_f_train,
                                  sgd_f_train, knn_f_train,
                                  X_train, lda_train], axis=1)

    new_x_test = np.concatenate([rf_f_test, et_f_test,
                                 gb_f_test, svc_f_test,
                                 ova_svm_f_test, mnb_f_test,
                                 sgd_f_test, knn_f_test,
                                 X_test, lda_test], axis=1)

    np.save('./dataset/new_x_train.npy', new_x_train)
    np.save('./dataset/new_x_test.npy', new_x_test)

rf2 = SklearnHelper(clf=RandomForestClassifier, seed=SEED, params=rf_params)
et2 = SklearnHelper(clf=ExtraTreesClassifier, seed=SEED, params=et_params)
xgb2 = SklearnHelper(clf=xgb.XGBClassifier, seed=SEED, params=xgb_params)
mnb2 = SklearnHelper(clf=MultinomialNB, seed=-1, params=mnb_params)
ova_svm2 = SklearnHelper(clf=OneVsRestClassifier, seed=-1, params=ova_params)

rf2_oof_train, rf2_oof_test = get_oof(rf2, new_x_train, Y_train, new_x_test)  # Random Forest
et2_oof_train, et2_oof_test = get_oof(et2, new_x_train, Y_train, new_x_test)  # Extra Trees
xgb2_oof_train, xgb2_oof_test = get_oof(xgb2, new_x_train, Y_train, new_x_test)  # Extra Trees
mnb2_oof_train, mnb2_oof_test = get_oof(mnb2, new_x_train, Y_train, new_x_test)
ova_svm2_oof_train, ova_svm2_oof_test = get_oof(ova_svm2, new_x_train, Y_train, new_x_test)

print('final feature importance')
x_train = np.concatenate((et_oof_train,
                          rf_oof_train,
                          gb_oof_train,
                          svc_oof_train,

                          ova_svm_oof_train,
                          mnb_oof_train,
                          sgd_oof_train,
                          knn_oof_train,

                          mnb2_oof_train,
                          ova_svm2_oof_train,

                          rf2_oof_train,
                          et2_oof_train,
                          xgb2_oof_train), axis=1)
print(x_train.shape)
x_test = np.concatenate((et_oof_test,
                         rf_oof_test,
                         gb_oof_test,
                         svc_oof_test,

                         ova_svm_oof_test,
                         mnb_oof_test,
                         sgd_oof_test,
                         knn_oof_test,

                         mnb2_oof_test,
                         ova_svm2_oof_test,

                         rf2_oof_test,
                         et2_oof_test,
                         xgb2_oof_test), axis=1)
print(x_test.shape)

gbm = xgb.XGBClassifier(
    # learning_rate = 0.02,
    n_estimators=2000,
    max_depth=4,
    min_child_weight=2,
    gamma=0.9,
    subsample=0.8,
    colsample_bytree=0.8,
    objective='binary:logistic',
    nthread=-1,
    scale_pos_weight=1).fit(x_train, Y_train)

# predictions =
y_true, y_pred = Y_test, gbm.predict(x_test)
print(classification_report(y_true, y_pred))

y_pred = lb.inverse_transform(y_pred)

DataHelper.save_submission('ensemble', y_pred)
