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
    def make_pairs(bins, label, paper_authors):
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
                unknown = None
                for j in range(len(bins[author_ix][0])):
                    unknown = rand.choice(bins[author_ix][0])
                    if b[1] not in paper_authors[unknown]:
                        break
                    if j == len(bins[author_ix][0]) - 1:
                        raise RuntimeError('all papers in co authoship')
                yes_no.append((unknown, b[1]))
                bins[author_ix][0].remove(unknown)
            else:
                raise ValueError('label is always either YES or NO')
        return yes_no, bins

    @staticmethod
    def write_pairs(filename, label, yes_no, bins, params='w'):
        if label is not 'YES' and label is not 'NO':
            raise ValueError('Label must always be YES or NO')

        pairs = []
        with open(filename, params) as of:
            for unknown, author in yes_no:
                for b in bins:
                    for paper in b[0]:
                        if b[1] == author:
                            assert (unknown != paper)
                            pairs.append((author, paper, unknown, label))
                            of.write(author + ',' + paper + ',unknown,' + unknown + ',' + label + '\n')
            return pairs

    @staticmethod
    def read_input(label, infile='labels.csv'):
        with open(infile) as fin:
            paper_authors = {}
            #for root, dirs, files in os.walk('./' + label):
            #    if root != '.':
            #        authors.append(str(root.lstrip('./')))
            csv_reader = csv.reader(fin, delimiter=',')
            authors = csv_reader.next()[1:]#skip header
            lines = []

            for line in csv_reader:
                paper = line[0]
                authorship = line[1:]
                tmp = []
                for i, value in enumerate(authorship):
                    if value == '1':
                        tmp.append(authors[i])
                paper_authors[paper] =  tmp
                lines.append(paper)

        assert (len(authors) == 20)
        assert (len(paper_authors) == 400)
        assert (len(lines) == 400)
        return authors, paper_authors, lines

    @staticmethod
    def tr_tst_val_split(authors, paper_authors, lines, ntr=14, ntst=4):
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
            train, val = select.train_test_split(train, train_size=ntr)
            train = rand.sample(train, 14)#shuffle papers
            test = rand.sample(test, 4)
            val = rand.sample(val, 2)
            tr_bins.append((train, author, paper_authors))
            tst_bins.append((test, author, paper_authors))
            val_bins.append((val, author, paper_authors))
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
        yes_no_pairs = []

        for i ,label in enumerate(labels):
            p = 'w'

            authors, paper_authors, lines = MLP400AV_API.read_input(label=label)
            tr_bins, tst_bins, val_bins = MLP400AV_API.tr_tst_val_split(authors, paper_authors, lines)
            yes_no, bins = MLP400AV_API.make_pairs(bins=tr_bins, label=label, paper_authors=paper_authors)
            tr_pairs = MLP400AV_API.write_pairs(filename=label + 'train.csv', label=label, yes_no=yes_no, bins=bins, params=p)
            yes_no, bins = MLP400AV_API.make_pairs(bins=tst_bins, label=label, paper_authors=paper_authors)
            test_pairs = MLP400AV_API.write_pairs(filename=label + 'test.csv', label=label, yes_no=yes_no, bins=bins, params=p)
            yes_no, bins = MLP400AV_API.make_pairs(bins=val_bins, label=label, paper_authors=paper_authors)
            val_pairs = MLP400AV_API.write_pairs(filename=label + 'val.csv', label=label, yes_no=yes_no, bins=bins, params=p)
            yes_no_pairs.append((tr_pairs, test_pairs, val_pairs))

        return yes_no_pairs

def main():
    MLP400AV_API.create_dataset()


if __name__=="__main__":
    print('hello')
    main()
