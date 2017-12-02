import numpy as np
import spacy as spacy

import baselines.prepare as prepare


def spacy_doc2vec():
    nlp = spacy.load('en_vectors_web_lg')
    (train_data, y_train), (val_data, y_val), (test_data, y_test) = prepare.load_data_tuple()
    docs_tr = []
    docs_val = []
    docs_test = []
    assert len(train_data) % 2 == 0

    for i in range(len(train_data)):
        docs_tr.append(nlp(train_data[i]))
        docs_val.append(nlp(val_data[i]))
        docs_test.append(nlp(test_data[i]))

    return (docs_tr[0:len(docs_tr) / 2], docs_tr[len(docs_tr) / 2:len(docs_tr)], y_train), \
           (docs_val[0:len(docs_val) / 2], docs_val[len(docs_val) / 2:len(docs_val)], y_val), \
           (docs_test[0:len(docs_test) / 2], docs_test[len(docs_test) / 2:len(docs_test)], y_test)


def spacy_doc2vec_sim(X):
    y = np.zeros(len(X))
    for i, sample in enumerate(X):
        y[i] = sample[0].similarity(sample[1])
    return y


def main():
    train, val, test = spacy_doc2vec()
    print(spacy_doc2vec_sim(train[0]))


if __name__ == '__main__':
    main()
