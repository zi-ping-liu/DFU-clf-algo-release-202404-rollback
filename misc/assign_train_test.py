# Import libraries
import numpy as np
import pandas as pd
import random

df_BSV = pd.read_csv("/home/efs/ziping/workspaces/dfu/clf_algo_release_202404_rollback/src/data/WAUSI_unifiedv4_BSV_reverted_20240422.csv")
df_shiftwin = pd.read_csv("/home/efs/ziping/workspaces/dfu/clf_algo_release_202404_rollback/src/data/WAUSI_unifiedv4_BSV+slidingwindow_reverted_20240422.csv")

# Drop 'bad_quality' and 'exclude_from_classification' rows
df_BSV = df_BSV[~df_BSV['DS_split'].isin(['bad_quality', 'exclude_from_classification'])].reset_index(drop = True)
df_shiftwin = df_shiftwin[~df_shiftwin['DS_split'].isin(['bad_quality', 'exclude_from_classification'])].reset_index(drop = True)

# Assign all US subjects to train
df_BSV.loc[(df_BSV['USorUK'] == 'US'), 'DS_split'] = 'train'
df_shiftwin.loc[(df_shiftwin['USorUK'] == 'US'), 'DS_split'] = 'train'

# Randomly select n_sample healing and nonhealing US subjects
df_US = df_BSV[(df_BSV['USorUK'] == 'US')].reset_index(drop = True)
n_sample = 20

pos_df_US = df_US[(df_US['GT'] == 1)].reset_index(drop = True)
pos_subjs_US = pos_df_US['subject_number'].unique().tolist()
random.seed(42)
selected_pos_US = random.sample(pos_subjs_US, n_sample)

neg_df_US = df_US[(df_US['GT'] == 0)].reset_index(drop = True)
neg_subjs_US = neg_df_US['subject_number'].unique().tolist()
random.seed(42)
selected_neg_US = random.sample(neg_subjs_US, n_sample)

df_BSV.loc[df_BSV['subject_number'].isin(selected_pos_US + selected_neg_US), 'DS_split'] = 'test'
df_shiftwin.loc[df_shiftwin['subject_number'].isin(selected_pos_US + selected_neg_US), 'DS_split'] = 'test'

tot = len(df_shiftwin['subject_number'].unique())
print(f"TOT: {tot}")

us_tot = len(df_shiftwin[df_shiftwin['USorUK'] == 'US']['subject_number'].unique())
us_train = len(df_shiftwin[(df_shiftwin['USorUK'] == 'US') & (df_shiftwin['DS_split'] == 'train')]['subject_number'].unique())
us_test = len(df_shiftwin[(df_shiftwin['USorUK'] == 'US') & (df_shiftwin['DS_split'] == 'test')]['subject_number'].unique())

print(f"US tot: {us_tot} | train: {us_train} | test: {us_test}")

eu_tot = len(df_shiftwin[df_shiftwin['USorUK'] == 'UK']['subject_number'].unique())
eu_train = len(df_shiftwin[(df_shiftwin['USorUK'] == 'UK') & (df_shiftwin['DS_split'] == 'train')]['subject_number'].unique())
eu_test = len(df_shiftwin[(df_shiftwin['USorUK'] == 'UK') & (df_shiftwin['DS_split'] == 'test')]['subject_number'].unique())

print(f"EU tot: {eu_tot} | train: {eu_train} | test: {eu_test}")

df_BSV.to_csv("/home/efs/ziping/workspaces/dfu/clf_algo_release_202404_rollback/src/data/WAUSI_unifiedv5_BSV_reverted_newsplit_20240422.csv", index = False)
df_shiftwin.to_csv("/home/efs/ziping/workspaces/dfu/clf_algo_release_202404_rollback/src/data/WAUSI_unifiedv5_BSV+slidingwindow_reverted_newsplit_20240422.csv", index = False)