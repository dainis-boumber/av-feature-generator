### MLP-400AV dataset

Machine Learning Papers' Authorship (MPLA-400) dataset contains 20 publications by each of the top-20 authors in ML, for the total of 400.

The data is located in [av-feature-generator](https://github.com/dainis-boumber/av-feature-generator/tree/master/data/MLP-400AV) project's `./ml_dataset` directory. You can also obtain it as a tarball from

Format: csv

Author, Known_paper, Unknown_paper, Is_same_author

The data is split under to schemas: A and B. For both schemas, unknown samples is not present 
amongst the known ones, unlike in PAN (this can be changed easily)

Schema A: disjoint sets (Train, Val and Test do NOT intersect)

A balanced dataset, which is formed as follows:
 1) Papers for each author are shuffled
 2) Authors are shuffled
 3) From each, we sample 70% for training, 10% for validation and 20% for testing, forming 3 subsets
 4) Then by only using data from within each of the subsets from (2), we designate papers whose
   author is "unknown" 
   
 Therefore, while there is some intersection between the authors, none of the papers are present in more
 than one of subsets formed.
 
 The results are saved in Atest.csv, Aval.csv and Atrain.csv
 
Schema B: Train, Val and Test do MAY intersect

A balanced dataset, which is formed as follows:
 1) Papers for each author are shuffled
 2) Authors are shuffled
 3) For each author, we designate papers whose author is "unknown"
 2) From each, we sample 70% for training, 10% for validation and 20% for testing, forming 3 subsets
 3) Then by only using data from within each of the subsets from (2), we designate papers whose
   author is "unknown"
   
 Some of the papers present in one subset can be in the other, as well. For example a paper from Test
 can serve as a negative example in Train.
 
 
  The results are saved in Btest.csv, Bval.csv and Btrain.csv
   

`Atest.csv` test , 20% of the data
`Aval.csv` - validation, 10% of the data
`Atrain.csv` - traininig set, 70% of the data.

The dataset is produced using MLPA-400's
`Labels.csv` which contains the ground truths for the paper authorship in the following format: <filename>,<author_1>,<author2>...<author_20>\n
 <filename> is plain text and <author_n> is a digit 0 or 1 indicating whether this author is one of the co-authors. The first row is the header row.

See [MLPA-400](https://github.com/dainis-boumber/AA_CNN/wiki/MLPA-400-Dataset) for more details.