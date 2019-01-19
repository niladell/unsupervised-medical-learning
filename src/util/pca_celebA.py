import numpy as np
import pandas as pd


labelsDf = pd.read_csv('/Users/ines/Downloads/project_stuff/celeba-dataset/list_attr_celeba.csv')

labelsDf.replace(-1,0, inplace=True)
