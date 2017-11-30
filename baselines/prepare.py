from pathlib import Path
import pickle
import logging

import numpy as np
from scipy import sparse

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

import data.MLP400AV.mlpapi as mlpapi


def load(schema='A2', path_to_ds='../data/MLP400AV/'):
    loader = mlpapi.MLPVLoader(schema, fileformat='pandas', directory=path_to_ds)
    train, val, test = loader.get_mlpv()
    return train, val, test


def load_data_tuple():
    data_pickle = Path("av400tuple.pickle")
    if not data_pickle.exists():
        logging.info("loading data structure from RAW")
        train, val, test = load()
        train_data = (train[1], (train[3]))
        val_data = (val[1], val[3])
        test_data = (test[1], test[3])  # col 1 is known col 3 is unknown
        y_train = train[4].tolist()
        y_val = val[4].tolist()
        y_test = test[4].tolist()
        logging.info("load data structure completed")

        pickle.dump([train_data, val_data, test_data, y_train, y_val, y_test], open(data_pickle, mode="wb"))
        logging.info("dumped all data structure in " + str(data_pickle))
    else:
        logging.info("loading data structure from PICKLE")
        [train_data, val_data, test_data, y_train, y_val, y_test] = pickle.load(open(data_pickle, mode="rb"))
        logging.info("load data structure completed")

    return (train_data, y_train), (val_data, y_val), (test_data, y_test)


def transform_tuple(X_train, X_val, X_test, vectorizer:CountVectorizer):
    vectorizer.fit(X_train[0].append(X_train[1]))
    train = tuple(vectorizer.transform(x) for x in X_train)
    val = tuple(vectorizer.transform(x) for x in X_val)
    test = tuple(vectorizer.transform(x) for x in X_test)
    return train, val, test



def data_vector_sbs(vectorizer):
    (train_data, train_y), (val_data, val_y), (test_data, test_y) = load_data_tuple()
    train_vec, val_vec, test_vec = transform_tuple(train_data, val_data, test_data, vectorizer)
    train_vec = sparse.hstack((train_vec[0], train_vec[1])).tocsr()
    val_vec = sparse.hstack((val_vec[0], val_vec[1])).tocsr()
    test_vec = sparse.hstack((test_vec[0], test_vec[1])).tocsr()
    return (train_vec, train_y), (val_vec, val_y), (test_vec, test_y)


def data_vector_diff(vectorizer):
    (train_data, train_y), (val_data, val_y), (test_data, test_y) = load_data_tuple()
    train_vec, val_vec, test_vec = transform_tuple(train_data, val_data, test_data, vectorizer)
    train_vec = (train_vec[0] - train_vec[1]).tocsr()
    val_vec = (val_vec[0] - val_vec[1]).tocsr()
    test_vec = (test_vec[0] - test_vec[1]).tocsr()
    return (train_vec, train_y), (val_vec, val_y), (test_vec, test_y)


def main():
    pass

if __name__ == '__main__':
    main()
