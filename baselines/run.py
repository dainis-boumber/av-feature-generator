import prepare as prep
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import scorer
import numpy as np
from scipy import sparse

def main():
    (ktrain, utrain, ytrain), (kval, uval, yval),  (ktest, utest, ytest) = prep.one_hot()
    train = np.hstack((ktrain, utrain))
    val = sparse.hstack(kval, uval)
    test = np.hstack((ktest, utest))
    clf = LogisticRegression()
    clf.fit(ktrain, ytrain)
    pred = clf.predict(test)
    acc = scorer.accuracy_score(ytest, pred)
    print(acc)

if __name__=='__main__':
    main()