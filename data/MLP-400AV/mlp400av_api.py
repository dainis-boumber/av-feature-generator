import os
import csv
import random as rand
import numpy as np
import sklearn.model_selection as select

# API container, static methods only
class MLP400AV_API:

    def __init__(self):
        pass

    @staticmethod
    def make_pairs(bins, label):
        yes_no = []
        for i, b in enumerate(bins):
            if label == 'YES':
                unknown = rand.choice(b[0])
                yes_no.append((unknown, b[1]))
                b[0].remove(unknown)
            elif label == 'NO':
                idx = []
                idx.extend((range(0, 20)))
                idx.remove(i)
                author_ix = rand.choice(idx)
                unknown = rand.choice(bins[author_ix][0])
                yes_no.append((unknown, b[1]))
                bins[author_ix][0].remove(unknown)
            else:
                raise ValueError('label is always either YES or NO')
            return yes_no, bins

    @staticmethod
    def write_pairs(filename, label, yes_no, bins):
        if label is not 'YES' or label is not 'NO':
            raise ValueError('Label must always be YES or NO')

        with open(filename, 'w') as of:
            for unknown, author in yes_no:
                for b in bins:
                    for paper in b[0]:
                        if b[1] == author:
                            assert (unknown != paper)
                            of.write(author + ',' + paper + ',unknown,' + unknown + ',' + label + '\n')

    @staticmethod
    def read_input(label, infile='labels.csv'):
        with open(infile) as fin:
            authors = []
            for root, dirs, files in os.walk('./' + label):
                if root != '.':
                    authors.append(str(root.lstrip('./')))
            csv_reader = csv.reader(fin, delimiter=',')
            header = csv_reader.next()[1:]#skip header
            lines = []
            for line in csv_reader:
                lines.append(line[0])

        assert (len(authors) == 20)
        return authors, lines

    @staticmethod
    def tr_tst_val_split(authors, lines, ntr=14, ntst=4):
        i = 0
        j = len(authors)
        tr_bins = []
        tst_bins = []
        val_bins = []

        while j <= 400:
            selection = lines[i:j]
            author = None
            for a in authors:
                if a in selection[0]:
                    author = a
                    break

            train, test = select.train_test_split(selection, test_size=ntst)
            train, val = select.train_test_split(tr_bins, train_size=ntr)
            train = rand.sample(train, 14)#shuffle papers
            test = rand.sample(test, 4)
            val = rand.sample(val, 2)
            tr_bins.append((train, author))
            tst_bins.append((test, author))
            val_bins.append((val, author))
            i += len(selection)
            j += len(selection)
        # shuffle authors
        tr_bins = rand.sample(tr_bins, len(tr_bins))
        tst_bins = rand.sample(tst_bins, len(tst_bins))
        val_bins = rand.sample(val_bins, len(val_bins))

        return tr_bins, tst_bins, val_bins

    @staticmethod
    def create_dataset():
        labels = ['YES', 'NO']

        for label in labels:
            authors, lines = MLP400AV_API.read_input(label=label)
            tr_bins, tst_bins, val_bins = MLP400AV_API.tr_tst_val_split(authors, lines)
            yes_no, bins = MLP400AV_API.make_pairs(bins=tr_bins, label=label)
            MLP400AV_API.wrte_pairs('train.csv', label=label, yes_no=yes_no, bins=bins)
            yes_no, bins = MLP400AV_API.make_pairs(bins=tst_bins, label=label)
            MLP400AV_API.wrte_pairs('test.csv', label=label, yes_no=yes_no, bins=bins)
            yes_no, bins = MLP400AV_API.make_pairs(bins=val_bins, label=label)
            MLP400AV_API.wrte_pairs('val.csv', label=label, yes_no=yes_no, bins=bins)


def main():
    MLP400AV_API.create_dataset()


if __name__=="main":
    main()
