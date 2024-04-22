# Import libraries
import pandas as pd
import numpy as np

bsv = pd.read_csv("/home/efs/ziping/workspaces/dfu/clf_algo_release_202404_rollback/src/data/WAUSI_unifiedv4_BSV_reverted_20240422.csv")
# bsv = pd.read_csv("/home/efs/ziping/workspaces/dfu/clf_algo_release_202404_rollback/src/data/WAUSI_unifiedv4_BSV_revertedMinMax_20240422.csv")

shiftwin = pd.read_csv("/home/efs/ziping/workspaces/dfu/clf_algo_release_202404_rollback/src/data/WAUSI_unifiedv4_slidingwindow_reverted_20240422.csv")
# shiftwin = pd.read_csv("/home/efs/ziping/workspaces/dfu/clf_algo_release_202404_rollback/src/data/WAUSI_unifiedv4_slidingwindow_revertedMinMax_20240422.csv")
shiftwin = shiftwin[bsv.columns]

for col in shiftwin.columns:
    # Non-clinical features - Skip
    if col in ['ICGUID', 'subject_number', 'Site', 'DE_phase', 'DS_phase', 'DS_split', 'good_ori', 'GT', 'Wound Location', 'USorUK', 'Visit Number', 'orientation_deg']:
        continue
    
    # Missing values for these features shall be handled later during training
    if col in ['exu_type', 'exu_volume', 'cm3_volume', 'cm2_surf_area', 'cm2_planar_area', 'cm_bbox_x', 'cm_bbox_y', 'cm_bbox_z']:
        continue
    
    print(col)
    
    for idx in range(len(shiftwin)):
        
        subjectID = shiftwin.loc[idx, 'subject_number']
        curvisit = shiftwin.loc[idx, 'Visit Number']
        
        if pd.isna(shiftwin.loc[idx, col]):
            
            # Missing values are filled with mean value based on all visit for the same patient
            if col in ['height_inches', 'weight_pounds', 'temp', 'heart_rate', 'systolic_blood_pressure', 'diastolic_blood_pressure']:
                u = bsv[bsv['subject_number'] == subjectID][col].values.tolist()
                v = shiftwin[shiftwin['subject_number'] == subjectID][col].values.tolist()
                replaced = np.nanmean(u + v)
                shiftwin.loc[idx, col] = replaced
                print(f"    >> Subject {subjectID} (SV{curvisit[6:]}): replace with mean value - {replaced}")
                
            # Missing values are filled with BSV entry for the same patient
            else:
                replaced = bsv[bsv['subject_number'] == subjectID][col].drop_duplicates().values[0]
                shiftwin.loc[idx, col] = replaced
                print(f"    >> Subject {subjectID} (SV{curvisit[6:]}): replace with BSV value - {replaced}")
                
df = pd.concat([bsv, shiftwin], ignore_index = True)

df = df[df['ICGUID'].isin(
    ['ddc8d7a3-a324-4055-9385-7613c9560784', 
     '1647e48b-de74-4b48-b2bb-b79f1ac88d41', 
     '8b60ddc8-8b78-462e-93ce-9bab3af975cf', 
     '89654108-081a-4f7e-98ea-b119da1af3a5', 
     'c3372c7a-0b45-4c21-98ef-8f09ffee447b'])].reset_index(drop = True)

subjectIDs = df['subject_number'].unique()
for col in df.columns:
    if col in ['ICGUID', 'subject_number', 'Site', 'DE_phase', 'DS_phase', 'DS_split', 'good_ori', 'GT', 'Wound Location', 'USorUK', 'Visit Number', 'orientation_deg',
               'cm3_volume', 'cm2_surf_area', 'cm2_planar_area', 'cm_bbox_x', 'cm_bbox_y', 'cm_bbox_z']:
        continue
    print(col)
    for subjectID in subjectIDs:
        entries = df[df['subject_number'] == subjectID][col].unique()
        if len(entries) > 1:
            print(f"    - {subjectID}: {entries}")
            
for subject in df['subject_number'].unique():
    entries = df[df['subject_number'] == subject]['DS_split'].unique()
    if len(entries) > 1:
        print(f"{subject} - {entries}")

df.to_csv("/home/efs/ziping/workspaces/dfu/clf_algo_release_202404_rollback/src/data/WAUSI_unifiedv4_BSV+slidingwindow_reverted_20240422.csv", index = False)    
# df.to_csv("/home/efs/ziping/workspaces/dfu/clf_algo_release_202404_rollback/src/data/WAUSI_unifiedv4_BSV+slidingwindow_revertedMinMax_20240422.csv", index = False)