import os
import csv
import random as rand


with open('labels.csv') as fin:
    csv_reader = csv.reader(fin, delimiter=',')
    csv_reader.next()
    lines = []
    for line in csv_reader:
        lines.append(line)

    i = 0
    j = 20
    bins = []

    while j <= 400:
        bins.append(lines[i:j])
        i = i + 20
        j = j + 20

    yes = []
    no = []
    assert(len(bins) == 20)

    for i in range(len(bins)):
        yes.append(rand.choice(bins[i]))
        bins[i].remove(yes[-1])
        no.append(rand.choice(bins[i]))
        bins[i].remove(no[-1])
        assert(len(bins[i]) == 18)

    print(yes)
    print(no)
'''
list_of_files = {}
for (dirpath, dirnames, filenames) in os.walk(path):
    for filename in filenames:
        if filename.endswith('.txt'):
            list_of_files[filename] = os.sep.join([dirpath, filename])
'''
