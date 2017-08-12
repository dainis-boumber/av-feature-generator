import spacy
from textacy import preprocess
import os
from utils.WordDict import getListListPANAndFoldersPAN


class PAN:

    def __init__(self, year):
        p = os.path.abspath(__file__ + "/../../data/PAN" + str(year) + '/')
        ll = getListListPANAndFoldersPAN(p)

