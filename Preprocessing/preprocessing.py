## Preprocessing

## Dependencies
import numpy as np
import pandas as pd 
import scipy
from matplotlib import pyplot as plt
import seaborn as sns


train = pd.read_csv('../Data/insurance_claims.csv')

## 64 bit values to 32 bit values

def compressData(train):
    for col in train.columns.values:
        if train[col].dtype == 'int64':
            train[col] = np.array(train[col], dtype = np.int32)
        if train[col].dtype == 'float64':
            train[col] = np.array(train[col], dtype = np.float32)
    return train


train = compressData(train)

##target = insurance money given
train['target'] = train['charges'] * train['insuranceclaim']


train.to_csv('../Data/Compressed.csv')

print("File saved in Data Folder")