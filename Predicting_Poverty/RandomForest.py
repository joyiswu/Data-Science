import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestClassifier

# data directory
DATA_DIR = os.path.join('..', 'data', 'processed')




# Standardize features
def standardize(df, numeric_only=True):
    numeric = df.select_dtypes(include=['int64', 'float64'])

    # subtracy mean and divide by std
    df[numeric.columns] = (numeric - numeric.mean()) / numeric.std()

    return df


def pre_process_data(df, enforce_cols=None):
    print("Input shape:\t{}".format(df.shape))

    df = standardize(df)
    print("After standardization {}".format(df.shape))

    # create dummy variables for categoricals   使用get_dummies进行one-hot编码
    df = pd.get_dummies(df)
    print("After converting categoricals:\t{}".format(df.shape))

    # match test set and training set columns
    if enforce_cols is not None:
        # Find the set difference of two arrays.
        to_drop = np.setdiff1d(df.columns, enforce_cols)
        to_add = np.setdiff1d(enforce_cols, df.columns)

        df.drop(to_drop, axis=1, inplace=True)
        df = df.assign(**{c: 0 for c in to_add})

    df.fillna(0, inplace=True)

    return df


from sklearn.ensemble import RandomForestClassifier


def train_model(features, labels, **kwargs):
    numTree = np.linspace(10, 150, 15)
    K = 10
    kf = KFold(n_splits=K)
    accuracyList = []
    for i in numTree:
        # instantiate model
        print(i)
        num = int(i)
        model_test = RandomForestClassifier(n_estimators=num, random_state=0)
        accuracy = 0
        for train_indices, test_indices in kf.split(features):
            # print(train_indices)
            train_labels = labels[train_indices]
            test_labels = labels[test_indices[0:-1]]
            # print(train_labels.shape, test_labels.shape)

            test_features = features[test_indices[0]:test_indices[-1]]

            train_features = features.drop(features.index[test_indices[0]:test_indices[-1] + 1])
            # .drop(aX_train.index[0:5])
            # print(train_indices[819:825],train_indices.shape)

            # print(train_features.shape,test_features.shape)
            # train model
            model_test.fit(train_features, train_labels)

            # get a (not-very-useful) sense of performance
            accuracy = accuracy + model_test.score(test_features, test_labels)

        accuracyList.append(accuracy / 10)
    print(accuracyList)

    number_of_estimator = int(numTree[accuracyList.index(max(accuracyList))])
    print(number_of_estimator)
    # instantiate model
    model = RandomForestClassifier(n_estimators=number_of_estimator, oob_score=True, random_state=0)

    # train model
    model.fit(features, labels)

    # get a (not-very-useful) sense of performance
    accuracy = model.score(features, labels)

    print(f"In-sample accuracy: {accuracy:0.2%}")
    return model

# load training data
a_train = pd.read_csv('A_hhold_train.csv', index_col='id')
b_train = pd.read_csv('B_hhold_train.csv', index_col='id')
c_train = pd.read_csv('C_hhold_train.csv', index_col='id')

# load test data
a_test = pd.read_csv('A_hhold_test.csv', index_col='id')
b_test = pd.read_csv('B_hhold_test.csv', index_col='id')
c_test = pd.read_csv('C_hhold_test.csv', index_col='id')


print("Country A")
aX_train = pre_process_data(a_train.drop('poor', axis=1))
# 将多维数组降为一维
ay_train = np.ravel(a_train.poor)

print("\nCountry B")
bX_train = pre_process_data(b_train.drop('poor', axis=1))
by_train = np.ravel(b_train.poor)

print("\nCountry C")
cX_train = pre_process_data(c_train.drop('poor', axis=1))
cy_train = np.ravel(c_train.poor)

numTree = np.linspace(10,150,15)
K = 10
kf = KFold(n_splits=K)
accuracyList = np.zeros(len(numTree))
print(numTree)

model_a = train_model(aX_train, ay_train)