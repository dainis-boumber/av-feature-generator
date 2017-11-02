import os
import csv
import random as rand
import numpy as np


with open('labels.csv') as fin:
    authors = []
    for root, dirs, files in os.walk("."):
        if root != '.':
            authors.append(str(root.lstrip('./')))
    csv_reader = csv.reader(fin, delimiter=',')
    header = csv_reader.next()[1:]
    lines = []
    for line in csv_reader:
        lines.append(line[0])

    i = 0
    j = 20
    assert (len(authors) == 20)
    tbins = []

    while j <= 400:
        selection = lines[i:j]
        author = None
        for a in authors:
            if a in selection[0]:
                author = a
                break


        tbins.append((lines[i:j], author))
        i = i + 20
        j = j + 20


    bins = rand.sample(tbins, 20)#shuffle
    #bins = tbins
    yes = []
    no = []
    assert(len(bins) == 20)

    for i, b in enumerate(bins):
        idx = []
        idx.extend((range(0, 20)))
        idx.remove(i)
        author_ix = rand.choice(idx)
        docs = bins[author_ix][0]
        unknown = rand.choice(docs)
        no.append((unknown,b[1]))
        bins[author_ix][0].remove(unknown)
        #print(unknown)
        #print(bins[author_ix][0])

    with open('no.csv', 'w') as of:
        for unknown, author in no:
            for b in bins:
                if b[1] == author:
                    for paper in b[0]:
                        of.write(author + ',' + paper + ',unknown,' + unknown + ',NO\n')

