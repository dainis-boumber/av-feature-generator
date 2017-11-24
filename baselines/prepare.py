import data.MLP400AV.mlpapi as mlpapi
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
import textacy as tx
import numpy as np
from scipy import sparse


def load(schema='A2', path_to_ds='../data/MLP400AV/'):
    loader = mlpapi.MLPVLoader(schema, fileformat='pandas', directory=path_to_ds)
    train, val, test = loader.get_mlpv()
    return train, val, test


def transform(X_train, X_val, X_test, vectorizer):
    vectorizer.fit(X_train)
    train = vectorizer.transform(X_train)
    val = vectorizer.transform(X_val)
    test = vectorizer.transform(X_test)
    return train, val, test


# bad unicode fix function clean('../data/MLP-400AV/YES/yee_whye_teh/yee_whye_teh_2_1.txt')
def clean(fname):
    text = tx.fileio.read.read_file(fname)
    text = tx.preprocess_text(fix_unicode=True, no_accents=True, transliterate=True, text=text)
    text = tx.preprocess.fix_bad_unicode(text=text)
    text = text.replace('?', 'f')
    with open(fname + '.txt', 'w') as f:
        f.write(text)


def one_hot():
    train, val, test = load()
    train_data = train[1].append(train[3])
    val_data = val[1].append(val[3])
    test_data = test[1].append(test[3])  # col 1 is known col 3 is unknown
    train_t, val_t, test_t = transform(train_data, val_data, test_data, CountVectorizer(binary=True, analyzer='char'))
    ktrain = train_t[:train_t.shape[0] / 2, ]
    utrain = train_t[train_t.shape[0] / 2:train_t.shape[0], ]
    X_train = sparse.hstack((ktrain, utrain))
    y_train = train[4].tolist()
    kval = val_t[:val_t.shape[0] / 2, ]
    uval = val_t[val_t.shape[0] / 2:val_t.shape[0], ]
    X_val = sparse.hstack((kval, uval))
    y_val = val[4].tolist()
    ktest = test_t[:test_t.shape[0] / 2, ]
    utest = test_t[test_t.shape[0] / 2:test_t.shape[0], ]
    X_test = sparse.hstack((ktest, utest))
    y_test = test[4].tolist()
    return (X_train, y_train), (X_val, y_val), (X_test, y_test)


def main():
    train, val, test = one_hot()


if __name__ == '__main__':
    main()
