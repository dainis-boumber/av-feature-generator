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
    data = np.row_stack((train, val, test))
    X = data[:,0:1].flatten()
    vcr = CountVectorizer(binary=True)
    vcr.fit_transform(X)

def clean(fname):
    text = tx.fileio.read.read_file(fname)
    text = tx.preprocess_text(fix_unicode=True, no_accents=True, transliterate=True, text=text)
    text = tx.preprocess.fix_bad_unicode(text=text)
    text = text.replace('?', 'f')
    with open(fname + '.txt', 'w') as f:
        f.write(text)

def main():
    load()
    #clean('../data/MLP-400AV/YES/yee_whye_teh/yee_whye_teh_2_1.txt')


if __name__=='__main__':
    main()