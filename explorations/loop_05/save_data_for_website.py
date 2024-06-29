import scipy.io
import pandas as pd
import os

# Create output directory if it doesn't exist
output_dir = 'data_for_website'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Load MATLAB files
filtered_QHD = scipy.io.loadmat('.filtered_QHD_table.mat')['filtered_QHD_table']
filtered_QDP = scipy.io.loadmat('filtered_QDP_table.mat')['filtered_QDP_table']
deleted_QHD = scipy.io.loadmat('deleted_QHD_table.mat')['deleted_QHD_table']
deleted_QDP = scipy.io.loadmat('deleted_QDP_table.mat')['deleted_QDP_table']

# Extract data and save as CSV files
filtered_QHD_df = pd.DataFrame(filtered_QHD)
filtered_QDP_df = pd.DataFrame(filtered_QDP)
deleted_QHD_df = pd.DataFrame(deleted_QHD)
deleted_QDP_df = pd.DataFrame(deleted_QDP)

filtered_QHD_df.to_csv(os.path.join(output_dir, 'filtered_QHD_table.csv'), index=False)
filtered_QDP_df.to_csv(os.path.join(output_dir, 'filtered_QDP_table.csv'), index=False)
deleted_QHD_df.to_csv(os.path.join(output_dir, 'deleted_QHD_table.csv'), index=False)
deleted_QDP_df.to_csv(os.path.join(output_dir, 'deleted_QDP_table.csv'), index=False)
