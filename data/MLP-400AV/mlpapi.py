import os
import csv
import random as rand
import sklearn.model_selection as select
import spacy

# API container, static methods only
class MLPAPI:

    AUTHORS = (
        'geoffrey_hinton',
        'vapnik',
        'bernard_scholkopf',
        'thomas_l_griffiths',
        'yann_lecun',
        'xiaojin_zhu',
        'yee_whye_teh',
        'radford_neal',
        'david_blei',
        'alex_smola',
        'michael_jordan',
        'zoubin_ghahramani',
        'daphne_koller',
        'lawrence_saul',
        'trevor_hastie',
        'thorsten_joachims',
        'yoshua_bengio',
        'andrew_y_ng',
        'tom_mitchell',
        'robert_tibshirani'
        )


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
    def write_pairs(filename, label, yes_no, bins):
        if label is not 'YES' and label is not 'NO':
            raise ValueError('Label must always be YES or NO')

        pairs = []
        with open(filename, 'w') as of:
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
            authors, paper_authors, lines = MLPAPI.read_input(label=label)
            tr_bins, tst_bins, val_bins = MLPAPI.tr_tst_val_split(authors, paper_authors, lines)
            yes_no, bins = MLPAPI.make_pairs(bins=tr_bins, label=label, paper_authors=paper_authors)
            tr_pairs = MLPAPI.write_pairs(filename=label + 'train.csv', label=label, yes_no=yes_no, bins=bins)
            yes_no, bins = MLPAPI.make_pairs(bins=tst_bins, label=label, paper_authors=paper_authors)
            test_pairs = MLPAPI.write_pairs(filename=label + 'test.csv', label=label, yes_no=yes_no, bins=bins)
            yes_no, bins = MLPAPI.make_pairs(bins=val_bins, label=label, paper_authors=paper_authors)
            val_pairs = MLPAPI.write_pairs(filename=label + 'val.csv', label=label, yes_no=yes_no, bins=bins)
            yes_no_pairs.append((tr_pairs, test_pairs, val_pairs))

        return yes_no_pairs

    @staticmethod
    def load_dataset(path_train='train.csv', path_test='test.csv', path_val='val.csv'):
        train_test_val_paths = (path_train, path_test, path_val)
        train = []
        test = []
        val = []
        for path in train_test_val_paths:
            with open(path) as pt:
                reader = csv.reader(pt)
                reader.next()
                for row in reader:
                    label = row[4]
                    k = open('./' + label + '/' + row[0] + '/' + row[1]).read()
                    for author in MLPAPI.AUTHORS:
                        if author in row[3]:
                            break
                    u = open('./' + label + '/' + author + '/' + row[3]).read()

                    if 'train' in path:
                        train.append((k, u, label))
                    elif 'test' in path:
                        test.append((k, u, label))
                    elif 'val' in path:
                        val.append((k, u, label))
                    else:
                        raise ValueError
        train = rand.sample(train, len(train))
        test = rand.sample(test, len(test))
        val = rand.sample(val, len(val))

        return train, val, test


class MLPVLoader:

    def __init__(self):
        self.train, self.val, self.test = MLPAPI.load_dataset()


    def get_mlpv(self):
        return self.train, self.val, self.test

    def slice(self, data, npieces=5):
        return [data[i:i + npieces] for i in range(0, len(data), npieces)]

    def get_slices(self):
        self.slice(self.train), self.slice(self.val), self.slice(self.test)

def main():
    loader=MLPVLoader()
    tr, v, tst = loader.get_slices()
    #MLP400AV_API.create_dataset()
    #tr, v, tst = MLPAPI.load_dataset()
    print('done')

if __name__=="__main__":
    print('hello')
    main()
