# Import libraries
import pandas as pd
import numpy as np
import json
import glob
import os



root_dir = "/home/efs/ziping/workspaces/dfu/clf_algo_release_202404_rollback/src/data/unified_csv_prep_data"

# retrieve Excel sheet for MSI data quality review
img_review = pd.read_excel(f"{root_dir}/MSI_review_SW_240419.xlsx")

# retrieve clinical features from Castor
wausi_us = pd.read_csv(f"{root_dir}/20240418/wausi_US_all_visits.csv")
wausi_eu = pd.read_csv(f"{root_dir}/20240418/wausi_EU_all_visits.csv")
clin_features = pd.concat([wausi_us, wausi_eu], ignore_index = True)
def get_uuid(row):
    return f"{row['subject_number']}_{(row['Visit Number']).split('_')[-1]}"
clin_features['uuid'] = clin_features.apply(get_uuid, axis = 1)

# merge above two dfs, rename and drop intermediate columns
sw_df = pd.merge(img_review, clin_features, on = 'uuid', how = 'left')
sw_df.rename(columns = {'subject_number_x': 'subject_number'}, inplace = True)
sw_df = sw_df[sw_df['debridement'] != 'Pre']
sw_df = sw_df[sw_df['v4_select'].str.strip() == 'Y']
sw_df = sw_df.drop(['VisitTime', 'start_SV', 'task assign', 'v4_select', 'Comment', 'subject_number_y', 'dfu_location'], axis = 1)

# retrieve 3D measurements
measurementsall_df = pd.read_csv(f"{root_dir}/WAUSI_measurements_updated.csv") # template used to clean up original 3D measurements data
measurementsNEW_df = pd.read_csv(f"{root_dir}/out_16.csv")
measurements_df = pd.concat([measurementsall_df, measurementsNEW_df]).drop_duplicates(['ICGUID'], keep = 'last').sort_values('ICGUID')

# merge
sw_df = pd.merge(sw_df, measurements_df, on = 'ICGUID', how = 'left')

# retrieve subject info from DE team
info_df = pd.read_excel(f"{root_dir}/20240322_WAUSI_all_potentially_usable_subjects_snapshot_COPY20240416.xlsx")
info_df.rename(columns={'DS_Phase ': 'DS_phase'}, inplace=True)

# Merge
sw_df = pd.merge(sw_df, info_df[['subject_number', 'DS_phase', 'Wound Location', 'USorUK']], on = 'subject_number', how = 'left')
df_all = sw_df.copy()

# limit orientation comparison to good_ori images
df = df_all[df_all['good_ori'] == 'Y'].reset_index(drop = True)
df_3d = df[['ICGUID', 'uuid', 'cm3_volume', 'cm2_surf_area', 'cm2_planar_area', 'cm_bbox_x', 'cm_bbox_y', 'cm_bbox_z', 'orientation_deg']]
# create a column for min orientation deg
df_3d['min_orientation_deg'] = df_3d.groupby('uuid')['orientation_deg'].transform('min')
# create the column to indicate the image is best orientation or not
df_3d['is_best'] = (df_3d['orientation_deg'] == df_3d['min_orientation_deg']).astype(int)
# create the df with best orientation image only
df_3d_best = df_3d[df_3d["is_best"] == 1]

# read the unified csv and drop the original 3d features
df_without_3d = df_all.drop(columns = ['cm3_volume', 'cm2_surf_area', 'cm2_planar_area', 'cm_bbox_x', 'cm_bbox_y', 'cm_bbox_z'])

# merge the df_3d_best with the unified csv based on ICGUID
final_3d_df = pd.merge(df_without_3d, df_3d_best, on = 'uuid', how = 'left')
final_3d_df.rename(columns = {'ICGUID_x': 'ICGUID'}, inplace = True)
final_3d_df.rename(columns = {'orientation_deg_x': 'orientation_deg'}, inplace = True)
final_3d_df = final_3d_df.drop(columns = ['ICGUID_y', 'orientation_deg_y', 'is_best', 'min_orientation_deg'])
final_3d_df['DS_split'] = 'train'
final_3d_df.drop(columns = ['uuid', 'debridement'], inplace = True)

# Exclude subjects
final_3d_df = final_3d_df[~final_3d_df['subject_number'].isin(['203-009', '202-034', '202-012', '202-035', '203-017'])].reset_index(drop = True)

final_3d_df.to_csv("/home/efs/ziping/workspaces/dfu/clf_algo_release_202404_rollback/src/data/WAUSI_unifiedv4_slidingwindow_reverted_20240422.csv", index = False)

final_3d_df.rename(columns={'ulcer_length': 'castor_length',
                            'ulcer_width': 'castor_width',
                            'cm_bbox_x': 'bbox_x',
                            'cm_bbox_y': 'bbox_y'}, inplace = True)
final_3d_df['ulcer_length'] = final_3d_df[['castor_length', 'castor_width']].max(axis = 1)
final_3d_df['ulcer_width'] = final_3d_df[['castor_length', 'castor_width']].min(axis = 1)
final_3d_df['cm_bbox_x'] = final_3d_df[['bbox_x', 'bbox_y']].max(axis = 1)
final_3d_df['cm_bbox_y'] = final_3d_df[['bbox_x', 'bbox_y']].min(axis = 1)
final_3d_df = final_3d_df.drop(columns = ['castor_length', 'castor_width', 'bbox_x', 'bbox_y'])

final_3d_df.to_csv("/home/efs/ziping/workspaces/dfu/clf_algo_release_202404_rollback/src/data/WAUSI_unifiedv4_slidingwindow_revertedMinMax_20240422.csv", index = False)