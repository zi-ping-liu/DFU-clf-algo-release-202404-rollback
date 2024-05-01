import pandas as pd

df = pd.read_csv("/home/efs/ziping/workspaces/dfu/clf_algo_release_202404_rollback/src/data/WAUSI_unifiedv7_BSV+slidingwindow_reverted_randsplit_1_20240501.csv")

feature_list = ['subject_number', 'Visit Number', 'age', 'sex', 'race', 'smoking', 'ckd', 'dialysis', 'dialysis_years', 'anemia', 'cirrhosis', 'DVT', 'heart_attack', 'stroke',
 'temp', 'heart_rate', 'systolic_blood_pressure', 'diastolic_blood_pressure', 'height_inches', 'weight_pounds', 'dfu_position', 'ulcer_duration', 'tendon_bone',
 'first_dfu_age', 'additional_DFUs', 'diabetic_neuropathy', 'skeletal_deformity', 'skeletal_study', 'limb_loss_history', 'limb_loss_study_foot_binary',
 'angio_stent_bypass', 'debridement_performed', 'exu_volume', 'exu_type', 'infection_binary']

df = df[feature_list]

subject_list = ['201-001', '201-003', '201-005', '201-006', '201-007', '201-008',
                '201-009', '201-010', '201-011', '201-012', '201-013', '201-014',
                '201-015', '201-016', '201-017', '201-023', '201-024', '201-025',
                '201-027', '201-029', '201-030', '201-032', '201-033', '201-035',
                '202-002', '202-004', '202-008', '202-009', '202-013', '202-014',
                '202-016', '202-017', '202-018', '202-019', '202-020', '202-021',
                '202-023', '202-024', '202-025', '202-028', '202-029', '202-030',
                '202-031', '202-032', '202-035', '202-037', '202-040', '202-046',
                '202-047', '202-050', '202-051', '202-054', '202-055', '202-057',
                '202-062', '202-063', '202-072', '202-076', '202-081', '202-083',
                '203-001', '203-002', '203-004', '203-005', '203-006', '203-010',
                '203-011', '203-012', '203-013', '203-015', '203-018', '203-020',
                '203-021', '203-024', '203-025', '203-028', '203-029', '203-031',
                '203-033', '203-034', '203-036', '203-037', '203-038', '203-039',
                '203-041', '203-043', '203-044', '203-045', '203-046', '203-047',
                '203-048', '203-050', '203-051', '203-052', '203-054', '203-057',
                '203-060', '203-061', '203-066', '203-067', '203-069', '203-074',
                '203-082', '203-083', '203-084', '203-088', '203-089', '203-090',
                '203-091', '205-001', '201-002', '201-021', '202-074', '202-077',
                '202-079', '203-003', '202-007', '202-036', '203-056', '203-058',
                '203-064', '203-068', '203-071', '203-076', '203-077', '203-079',
                '203-080', '203-022', '203-026', '203-055']

# 201-001
# 201-006
# 201-009
# 201-012
# 201-017
# 201-027
# 201-035
# 202-009
# 202-018
# 202-023
# 202-037
# 202-051
# 202-076
# 203-010
# 203-025
# 203-050
# 203-064
# 203-089
row = df[df['subject_number'] == '203-089'].drop_duplicates()
print(row['Visit Number'])
unique_feat_list = ['subject_number', 'age', 'sex', 'race', 'smoking', 'ckd', 'dialysis', 'dialysis_years', 
                    'anemia', 'cirrhosis', 'DVT', 'heart_attack', 'stroke', 'dfu_position', 'ulcer_duration', 
                    'tendon_bone', 'first_dfu_age', 'additional_DFUs', 'diabetic_neuropathy', 'skeletal_deformity', 
                    'skeletal_study', 'limb_loss_history', 'limb_loss_study_foot_binary', 'angio_stent_bypass']
assert(len(row[unique_feat_list].drop_duplicates()) == 1)

row = row[row['Visit Number'] == 'DFU_SV6']

for col in feature_list:
    if (row['Visit Number'].values[0] != 'DFU_SV1') & (col in unique_feat_list):
        continue        
    print(f"{col}:  {row[col].item()}")
