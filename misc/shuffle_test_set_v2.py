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

### Sample size
n_sample = 10

final_test_set = ['202-010', '202-078', '202-052', '202-049', '201-022', '203-030', '201-018', '202-044', '202-011', '202-053', '202-073', '202-070', '205-003', '202-067', 
                  '202-022', '202-069', '202-056', '202-064', '203-086', '202-041', '202-068', '203-072', '201-019', '202-042', '202-082', '203-065', '203-087', '201-028', 
                  '203-053', '201-031', '202-080', '203-085', '203-070', '202-048', '203-014', '201-034', '202-039', '202-043', '205-002', '202-075']

orig_df_sw = pd.read_csv("/home/efs/ziping/workspaces/dfu/clf_algo_release_202404_rollback/src/data/WAUSI_unifiedv6_BSV+slidingwindow_reverted_randsplit_5_20240430.csv")

# Combine final test subjects with all EU subjects
final_test_data = orig_df_sw[orig_df_sw['subject_number'].isin(final_test_set)].reset_index(drop = True)
eu_data = orig_df_sw[orig_df_sw['USorUK'] == 'UK'].reset_index(drop = True)
final_df = pd.concat([final_test_data, eu_data]).reset_index(drop = True)
final_df['DS_split'] = 'test' # set 'DS_split' to 'test' for all rows
# final_df.to_csv("/home/efs/ziping/workspaces/dfu/clf_algo_release_202404_rollback/src/data/WAUSI_unifiedv7_BSV+slidingwindow_reverted_final_test_20240501.csv", index = False)

# Delete 40 reserved test subjects from unified CSV
df_sw = orig_df_sw[~orig_df_sw['subject_number'].isin(final_test_set)].reset_index(drop = True)
# Delete EU subjects from unified CSV
df_sw = df_sw[df_sw['USorUK'] == 'US'].reset_index(drop = True)
# Set 'DS_split' to 'train' for all rows
df_sw['DS_split'] = 'train'

# Randomly select n_sample healing and nonhealing US subjects
pos_subjs_US = df_sw[(df_sw['Visit Number'] == 'DFU_SV1')& (df_sw['GT'] == 1)]['subject_number'].unique().tolist()
random.seed(seed)
selected_pos_US = random.sample(pos_subjs_US, n_sample)
neg_subjs_US = df_sw[(df_sw['Visit Number'] == 'DFU_SV1')& (df_sw['GT'] == 0)]['subject_number'].unique().tolist()
random.seed(seed)
selected_neg_US = random.sample(neg_subjs_US, n_sample)

df_sw.loc[df_sw['subject_number'].isin(selected_pos_US + selected_neg_US), 'DS_split'] = 'test'

### Sanity check
print(selected_pos_US + selected_neg_US)
tot = len(df_sw['subject_number'].unique())
print(f"TOT: {tot}")
us_tot = len(df_sw[df_sw['USorUK'] == 'US']['subject_number'].unique())
us_train = len(df_sw[(df_sw['USorUK'] == 'US') & (df_sw['DS_split'] == 'train')]['subject_number'].unique())
us_test = len(df_sw[(df_sw['USorUK'] == 'US') & (df_sw['DS_split'] == 'test')]['subject_number'].unique())

print(f"US tot: {us_tot} | train: {us_train} | test: {us_test}")

# df_sw.to_csv("/home/efs/ziping/workspaces/dfu/clf_algo_release_202404_rollback/src/data/WAUSI_unifiedv7_BSV+slidingwindow_reverted_randsplit_1_20240501.csv", index = False)
# df_sw.to_csv("/home/efs/ziping/workspaces/dfu/clf_algo_release_202404_rollback/src/data/WAUSI_unifiedv7_BSV+slidingwindow_reverted_randsplit_2_20240501.csv", index = False)
# df_sw.to_csv("/home/efs/ziping/workspaces/dfu/clf_algo_release_202404_rollback/src/data/WAUSI_unifiedv7_BSV+slidingwindow_reverted_randsplit_3_20240501.csv", index = False)
# df_sw.to_csv("/home/efs/ziping/workspaces/dfu/clf_algo_release_202404_rollback/src/data/WAUSI_unifiedv7_BSV+slidingwindow_reverted_randsplit_4_20240501.csv", index = False)
# df_sw.to_csv("/home/efs/ziping/workspaces/dfu/clf_algo_release_202404_rollback/src/data/WAUSI_unifiedv7_BSV+slidingwindow_reverted_randsplit_5_20240501.csv", index = False)