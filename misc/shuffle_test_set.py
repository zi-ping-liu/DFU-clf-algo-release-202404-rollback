# Import libraries
import numpy as np
import pandas as pd
import random



### Random seed
# seed = 75203
# seed = 75204
# seed = 75205
# seed = 75206
# seed = 75207
# Sample size
n_sample = 20


# Get subject list with SV1 available
df_BSV = pd.read_csv("/home/efs/ziping/workspaces/dfu/clf_algo_release_202404_rollback/src/data/WAUSI_unifiedv4_BSV_reverted_20240422.csv")
df_BSV = df_BSV[~df_BSV['DS_split'].isin(['bad_quality', 'exclude_from_classification'])].reset_index(drop = True)
df_BSV = df_BSV[(df_BSV['USorUK'] == 'US') & (df_BSV['good_ori'] == 'Y')].reset_index(drop = True)

# Randomly select n_sample healing and nonhealing US subjects
pos_subjs_US = df_BSV[df_BSV['GT'] == 1]['subject_number'].unique().tolist()
random.seed(seed)
selected_pos_US = random.sample(pos_subjs_US, n_sample)
neg_subjs_US = df_BSV[df_BSV['GT'] == 0]['subject_number'].unique().tolist()
random.seed(seed)
selected_neg_US = random.sample(neg_subjs_US, n_sample)

# Load BSV+ShiftWin unified CSV
df_sw = pd.read_csv("/home/efs/ziping/workspaces/dfu/clf_algo_release_202404_rollback/src/data/WAUSI_unifiedv5_BSV+slidingwindow_reverted_newsplit_20240422_updated.csv")
# Reset all labels to 'train'
df_sw['DS_split'] = 'train'
# Assign all EU subjects into 'test'
df_sw.loc[(df_sw['USorUK'] == 'UK'), 'DS_split'] = 'test'
# Assign 'test' label
df_sw.loc[df_sw['subject_number'].isin(selected_pos_US + selected_neg_US), 'DS_split'] = 'test'

# Sanity check
print(selected_pos_US + selected_neg_US)
tot = len(df_sw['subject_number'].unique())
print(f"TOT: {tot}")

us_tot = len(df_sw[df_sw['USorUK'] == 'US']['subject_number'].unique())
us_train = len(df_sw[(df_sw['USorUK'] == 'US') & (df_sw['DS_split'] == 'train')]['subject_number'].unique())
us_test = len(df_sw[(df_sw['USorUK'] == 'US') & (df_sw['DS_split'] == 'test')]['subject_number'].unique())

print(f"US tot: {us_tot} | train: {us_train} | test: {us_test}")

eu_tot = len(df_sw[df_sw['USorUK'] == 'UK']['subject_number'].unique())
eu_train = len(df_sw[(df_sw['USorUK'] == 'UK') & (df_sw['DS_split'] == 'train')]['subject_number'].unique())
eu_test = len(df_sw[(df_sw['USorUK'] == 'UK') & (df_sw['DS_split'] == 'test')]['subject_number'].unique())

print(f"EU tot: {eu_tot} | train: {eu_train} | test: {eu_test}")

# df_sw.to_csv("/home/efs/ziping/workspaces/dfu/clf_algo_release_202404_rollback/src/data/WAUSI_unifiedv6_BSV+slidingwindow_reverted_randsplit_1_20240430.csv", index = False)
# df_sw.to_csv("/home/efs/ziping/workspaces/dfu/clf_algo_release_202404_rollback/src/data/WAUSI_unifiedv6_BSV+slidingwindow_reverted_randsplit_2_20240430.csv", index = False)
# df_sw.to_csv("/home/efs/ziping/workspaces/dfu/clf_algo_release_202404_rollback/src/data/WAUSI_unifiedv6_BSV+slidingwindow_reverted_randsplit_3_20240430.csv", index = False)
# df_sw.to_csv("/home/efs/ziping/workspaces/dfu/clf_algo_release_202404_rollback/src/data/WAUSI_unifiedv6_BSV+slidingwindow_reverted_randsplit_4_20240430.csv", index = False)
# df_sw.to_csv("/home/efs/ziping/workspaces/dfu/clf_algo_release_202404_rollback/src/data/WAUSI_unifiedv6_BSV+slidingwindow_reverted_randsplit_5_20240430.csv", index = False)