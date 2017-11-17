import prepare as prep
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import scorer
import numpy as np
from scipy import sparse

def main():
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = prep.one_hot()
    clf = LogisticRegression()
    clf.fit(X_train, y_train)
    pred = clf.predict(X_test)
    acc = scorer.accuracy_score(y_test, pred)
    print(acc)

if __name__=='__main__':
    main()