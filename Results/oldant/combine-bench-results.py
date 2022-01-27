import pandas as pd

d1 = pd.read_csv('spm-secondary/bench-results-spm-p1.csv')
d2 = pd.read_csv('spm-secondary/bench-results-spm-p2.csv')
d3 = pd.read_csv('spm-secondary/bench-results-spm-p3.csv')
data = pd.concat([d1, d2, d3], ignore_index=True)
data.to_csv('spm-secondary/bench-results-spm.csv')


# d1 = pd.read_csv('bench-results-pm-p1.csv')
# d2 = pd.read_csv('bench-results-pm-p2.csv')
# d3 = pd.read_csv('bench-results-pm-p3.csv')
#
# data = pd.concat([d1, d2, d3], ignore_index=True)
#
# data.to_csv('bench-results-pm.csv')
