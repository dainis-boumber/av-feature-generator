import baselines.prepare as prep
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import BernoulliNB
from sklearn.metrics import scorer


def main():
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = prep.one_hot()
    clfs = [LinearSVC(), BernoulliNB()]

    for clf in clfs:
        clf.fit(X_train, y_train)
        pred = clf.predict(X_test)
        acc = scorer.accuracy_score(y_test, pred)
        print(acc)


if __name__ == '__main__':
    main()

