import mlpapi as mlpapi
import numpy as np
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import CountVectorizer
import textacy as tx

def load(schema='A2'):
    loader = mlpapi.MLPVLoader(schema, '../data/MLP-400AV/')
    train, val, test = loader.get_mlpv()
    train = np.asarray(train)
    val = np.asarray(val)
    test = np.asarray(test)
    return train, val, test

def transform(train, val, test, vectorizer):
    vectorizer.fit(np.row_stack((train[:0], train[:1])))
    tr_known = vectorizer.transform(train[:0])
    tr_unknown = vectorizer.transform(train[:1])
    val_known = vectorizer.transform(val[:0])
    val_unknown = vectorizer.transform(val[:1])
    test_known = vectorizer.transform(test[:0])
    test_unknown = vectorizer.transform(test[:1])
    raise NotImplementedError #TODO

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