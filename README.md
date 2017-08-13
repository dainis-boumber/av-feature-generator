# av-feature-generator
Feature generator for Authorship Verification problems with pair-wise data

After preprocess.py initializes the dataset, the format for each split is like:

```
{ "EN001" : [{"known" : ["/absolute/path/to/known.txt"], "unknown" : ["/absolute/path/to/unknown.txt"], "label" : True/False]}  
```

More documentation can be found on [Project Wiki](https://github.com/dainis-boumber/av-feature-generator/wiki)
