import glob 
import os

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

np.random.seed(33)

# File must be run from root 'poseidon' directory
image_dir = './data/scisrs/sig_images/yolo_images_dataset'
os.chdir(image_dir)

base_filenames = []
for file in glob.glob("*.png"):
    base_filenames.append(file[:-4])

# Split into train/test/val sets
base_filenames_train, base_filenames_other = train_test_split(base_filenames, test_size=0.3)
base_filenames_val, base_filenames_test = train_test_split(base_filenames_other, test_size=0.5)

# Create training dataframe
image_filenames_train = [f+".png" for f in base_filenames_train]
annotation_filenames_train = [f+".txt" for f in base_filenames_train]
df_train = pd.DataFrame(list(zip(image_filenames_train,annotation_filenames_train)), columns = ['image_filename','annotation_filename'])

# Create validation dataframe
image_filenames_val = [f+".png" for f in base_filenames_val]
annotation_filenames_val = [f+".txt" for f in base_filenames_val]
df_val = pd.DataFrame(list(zip(image_filenames_val,annotation_filenames_val)), columns = ['image_filename','annotation_filename'])

# Create testing datframe
image_filenames_test = [f+".png" for f in base_filenames_test]
annotation_filenames_test = [f+".txt" for f in base_filenames_test]
df_test = pd.DataFrame(list(zip(image_filenames_test,annotation_filenames_test)), columns = ['image_filename','annotation_filename'])

# Save the dataframes as csv files
os.chdir('../../../../metadata/scisrs')
df_train.to_csv('train.csv',index=False)
df_val.to_csv('val.csv',index=False)
df_test.to_csv('test.csv',index=False)