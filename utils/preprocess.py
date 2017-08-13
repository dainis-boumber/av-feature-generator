import spacy
from textacy import preprocess
import os
from utils.WordDict import getListListPANAndFoldersPAN


class PAN:

    def __init__(self, year):
        p = os.path.abspath(__file__ + "/../../data/PAN" + str(year) + '/')
        pair_dirs, self.splits = getListListPANAndFoldersPAN(p)
        self.data = {}

        for pair_dir in pair_dirs:
            _ , tail = os.path.split(pair_dir)
            description = []
            if 'train' in pair_dirs:
                with open('truth.txt') as truth:
                    for line in truth:
                        description.append({"label" : line.strip().split()[1]})
            for path in os.listdir(pair_dir):
                if not os.path.isfile(path): raise TypeError
                pair = {}
                if 'unknown' in path:
                    pair.update({'unknown' : path})
                else:
                    pair.update({'known' : path})
                description.append(pair)
            self.data.update({tail : description})



        pass

