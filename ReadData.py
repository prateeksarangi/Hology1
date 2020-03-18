from pandas import DataFrame, read_csv
import matplotlib.pyplot as plt
import pandas as pd 
import os
from shutil import copyfile


file = 'training/crime/train_labels.csv'
df = pd.read_csv(file)
i = 1
df = df.groupby('class').apply(pd.DataFrame.sort_values, 'filename')


for dirname, _, filenames in os.walk('training/crime/train'):
    for filename in filenames:
        if ((df['class'] == 'Handgun') & (df['filename'] == filename)).any():
        	copyfile(os.path.join('/Users/ashwini/Hology1/training/crime/train', filename), os.path.join('/Users/ashwini/Hology1/training/crime/train/Handgun', filename))

for dirname, _, filenames in os.walk('training/crime/train'):
    for filename in filenames:
        if ((df['class'] == 'Knife') & (df['filename'] == filename)).any():
        	copyfile(os.path.join('/Users/ashwini/Hology1/training/crime/train', filename), os.path.join('/Users/ashwini/Hology1/training/crime/train/Knife', filename))

for dirname, _, filenames in os.walk('training/crime/train'):
	for filename in filenames:
	    if ((df['class'] == 'Person') & (df['filename'] == filename)).any():
	    	copyfile(os.path.join('/Users/ashwini/Hology1/training/crime/train', filename), os.path.join('/Users/ashwini/Hology1/training/crime/train/Person', filename))


file = 'training/crime/test_labels.csv'
df = pd.read_csv(file)
i = 1
df = df.groupby('class').apply(pd.DataFrame.sort_values, 'filename')


for dirname, _, filenames in os.walk('training/crime/test'):
    for filename in filenames:
        if ((df['class'] == 'Handgun') & (df['filename'] == filename)).any():
        	copyfile(os.path.join('/Users/ashwini/Hology1/training/crime/test', filename), os.path.join('/Users/ashwini/Hology1/training/crime/test/Handgun', filename))

for dirname, _, filenames in os.walk('training/crime/test'):
    for filename in filenames:
        if ((df['class'] == 'Knife') & (df['filename'] == filename)).any():
        	copyfile(os.path.join('/Users/ashwini/Hology1/training/crime/test', filename), os.path.join('/Users/ashwini/Hology1/training/crime/test/Knife', filename))

for dirname, _, filenames in os.walk('training/crime/test'):
	for filename in filenames:
	    if ((df['class'] == 'Person') & (df['filename'] == filename)).any():
	    	copyfile(os.path.join('/Users/ashwini/Hology1/training/crime/test', filename), os.path.join('/Users/ashwini/Hology1/training/crime/test/Person', filename))


