import pandas as pd
import os
import re

os.chdir('scripts')

df = pd.read_csv('reads.csv')


# Extracting healthy subject (all radiologists agree that there are no findings)
healthy = df[df.sum(axis=1)==0]  # returns rows where all entries are zero.
names = healthy.loc[:,'name']
names.to_csv('healthy.csv', index=False)

# Extracting those with (typically more obvious lesions, like) hematomas or hemorrhages
radiologists = ['R1:', 'R2:', 'R3:']
wanted_features = ['ICH', 'IPH', 'SDH', 'EDH']
columns = [(x+y) for x in radiologists for y in wanted_features]  # gets a list with all the columns we want
columns.append('name')

H = df.loc[:, df.columns.isin(columns)]
H = H[H.sum(axis=1)!=0]
names = H.loc[:,'name']
names.to_csv('hemorrhages.csv', index=False)