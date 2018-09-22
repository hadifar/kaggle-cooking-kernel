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
from sklearn.cross_validation import KFold
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier

from data_helper import DataHelper

train_feature, train_label, test_feature = DataHelper.get_tfidf_vectorize()

ntrain = train_feature.shape[0]
ntest = test_feature.shape[0]
NFOLDS = 5  # set folds for out-of-fold prediction
kf = KFold(ntrain, n_folds=NFOLDS, random_state=0)

lb = LabelEncoder()
train_label = lb.fit_transform(train_label)


class SklearnHelper(object):
    def __init__(self, clf, seed=0, params=None):
        # params['random_state'] = seed
        self.clf = clf(**params)

    def train(self, x_train, y_train):
        self.clf.fit(x_train, y_train)

    def predict(self, x):
        return self.clf.predict(x)

    def fit(self, x, y):
        return self.clf.fit(x, y)

    def feature_importances(self, x, y):
        print(self.clf.fit(x, y).feature_importances_)


def get_oof(clf, x_train, y_train, x_test):
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


mnb_params = {'alpha': 1.0}
sgd_params = {'n_jobs': -1}

X_train = train_feature[:9000]
Y_train = train_label[:9000]
X_test = train_feature[9000:]
Y_test = train_label[9000:]

del train_feature
del test_feature
del train_label

# clf1 = MultinomialNB().fit(X_train, Y_train)
# y_true, y_pred = Y_test, clf1.predict(X_test)
# print(classification_report(y_true, y_pred))
#
# clf2 = SGDClassifier().fit(X_train, Y_train)
#
# model1 = SelectFromModel(clf1, prefit=True)
# model2 = SelectFromModel(clf2, prefit=True)
#
# X1_new = model1.transform(X_train)
# X2_new = model2.transform(X_train)
#
# X1_test = model1.transform(X_test)
# X2_test = model2.transform(X_test)
#
# NEW_FEATURE = np.concatenate([X2_new.toarray(), X1_new.toarray(), X_train.toarray()], axis=1)
# NEW_TEST_FEATURE = np.concatenate([X1_test.toarray(), X2_test.toarray(), X_test.toarray()], axis=1)
#
# clf_new = MultinomialNB().fit(NEW_FEATURE, Y_train)
# y_true, y_pred = Y_test, clf_new.predict(NEW_TEST_FEATURE)
# print(classification_report(y_true, y_pred))


tuned_parameters = {'n_estimators': [100, 500, 1000], 'max_depth': [5, 10, 20, 30, 50, 100],
                    'max_features': ['sqrt'], 'min_samples_leaf': [2]}

scores = ['precision']

for score in scores:
    print("# Tuning hyper-parameters for %s" % score)
    print()

    clf = GridSearchCV(GradientBoostingClassifier(verbose=1),
                       tuned_parameters, cv=3, scoring='%s_macro' % score, verbose=1, n_jobs=-1)
    clf.fit(X_train, Y_train)

    print("Best parameters set found on development set:")
    print()
    print(clf.best_params_)
    print()
    print("Grid scores on development set:")
    print()
    means = clf.cv_results_['mean_test_score']
    stds = clf.cv_results_['std_test_score']
    for mean, std, params in zip(means, stds, clf.cv_results_['params']):
        print("%0.3f (+/-%0.03f) for %r"
              % (mean, std * 2, params))
    print()

    print("Detailed classification report:")
    print()
    print("The model is trained on the full development set.")
    print("The scores are computed on the full evaluation set.")
    print()
    y_true, y_pred = Y_test, clf.predict(X_test)
    print(classification_report(y_true, y_pred))
    print()
