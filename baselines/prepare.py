import data.MLP400AV.mlpapi as mlpapi
import numpy as np
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import CountVectorizer
import textacy as tx

def load(schema='A2', path_to_ds='../data/MLP400AV/'):
    loader = mlpapi.MLPVLoader(schema, path_to_ds)
    train, val, test = loader.get_mlpv()
    return train, val, test

def transform(X_train, X_val, X_test, vectorizer):
    vectorizer.fit(np.row_stack((X_train[:0], X_train[:1])))
    tr_known = vectorizer.transform(X_train[:0])
    tr_unknown = vectorizer.transform(X_train[:1])
    val_known = vectorizer.transform(X_val[:0])
    val_unknown = vectorizer.transform(X_val[:1])
    test_known = vectorizer.transform(X_test[:0])
    test_unknown = vectorizer.transform(X_test[:1])
    return np.hstack((tr_known, tr_unknown)), np.hstack((val_known, val_unknown)), np.hstack((test_known, test_unknown))

def clean(fname):
    text = tx.fileio.read.read_file(fname)
    text = tx.preprocess_text(fix_unicode=True, no_accents=True, transliterate=True, text=text)
    text = tx.preprocess.fix_bad_unicode(text=text)
    text = text.replace('?', 'f')
    with open(fname + '.txt', 'w') as f:
        f.write(text)

def main():
    train, val, test = load()
    transform(train, val, test, CountVectorizer(binary=True))
    #clean('../data/MLP-400AV/YES/yee_whye_teh/yee_whye_teh_2_1.txt')


if __name__=='__main__':
    main()