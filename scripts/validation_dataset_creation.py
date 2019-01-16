import pandas as pd
import os

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

# Extracting those with a calvarial fracture
fracture = ['Fracture', 'CalvarialFracture', 'OtherFracture']
columns = [(x+y) for x in radiologists for y in fracture]
columns.append('name')

F = df.loc[:, df.columns.isin(columns)]
F = F[F.sum(axis=1)>1] # sum has to be strictly greater than 1 to exclude possible isolated mistakes in labelling.
names = F.loc[:,'name']
names.to_csv('fractures.csv', index=False)