# Import libraries
import os, sys
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from models.baseline import BaseLine
from utils.dfu_dataset import DFUDataset
from torch.utils.data import DataLoader
from utils.preprocess import PreProcessing



class Config:
    def __init__(self, data):
        for key, value in data.items():
            setattr(self, key, value)



def predict(test_loader, model, device):
    
    model.eval()
    
    res_df = pd.DataFrame()
    
    with torch.no_grad():
        
        for _, (clin_features, images, mask, target, (subject_number, visit_number, icguid)) in enumerate(test_loader, start = 1):
            
            clin_features = clin_features.to(torch.float32).to(device)
            
            images = images.to(device) # shape: (batch_size, C, H, W)
    
            mask = mask.to(device)
            
            target = torch.squeeze(target).long().to(device)
            
            if target.ndim == 0: target = target.unsqueeze(0)
            
            output = model(clin_features, images, mask)
            
            pred = F.softmax(output, dim = 1) # convert to probabilities
            pred = pred[:, 1]
            
            cur_df = pd.DataFrame({
                'subject_number': list(subject_number),
                'Visit Number': list(visit_number),
                'ICGUID': list(icguid),
                'Pred_Proba': pred.cpu().numpy(),
                'GT': target.cpu().numpy()
            })
            
            res_df = pd.concat([res_df, cur_df], ignore_index = True)
         
    return res_df



if __name__ == "__main__":
    
    # # Load model weights
    # model_path = "/home/efs/ziping/workspaces/dfu/clf_algo_release_202404_rollback/results/reproduce_shiftwin/baseline/hs_571"
    # best_epoch = 14
    # config = np.load(f"{model_path}/model/config_CV.npy", allow_pickle = True).item()
    # config = Config(config)
    # model = BaseLine(config).to(0)
    # model.load_state_dict(torch.load(f"{model_path}/model/snapshot_fold_99/epoch_{best_epoch}.pth"))
    
    # Load model weights
    model_path = "/home/efs/ziping/workspaces/dfu/clf_algo_release_202404_rollback/results/reproduce_shiftwin_v2/baseline/hs_571"
    best_epoch = 14
    config = np.load(f"{model_path}/model/config_final_train.npy", allow_pickle = True).item()
    config = Config(config)
    model = BaseLine(config).to(0)
    model.load_state_dict(torch.load(f"{model_path}/model/snapshot_fold_99/epoch_{best_epoch}.pth"))
    
    # # Load model weights
    # model_path = "/home/efs/ziping/workspaces/dfu/clf_algo_release_202404_rollback/results/shiftwin_v3/random_split_1/baseline/hs_994"
    # best_epoch = 2
    # config = np.load(f"{model_path}/model/config_CV.npy", allow_pickle = True).item()
    # config = Config(config)
    # model = BaseLine(config).to(0)
    # model.load_state_dict(torch.load(f"{model_path}/model/snapshot_fold_99/epoch_{best_epoch}.pth"))
    
    # # Load model weights
    # model_path = "/home/efs/ziping/workspaces/dfu/clf_algo_release_202404_rollback/results/shiftwin_v4/random_split_5/baseline/hs_571"
    # best_epoch = 1
    # config = np.load(f"{model_path}/model/config_CV.npy", allow_pickle = True).item()
    # config = Config(config)
    # model = BaseLine(config).to(0)
    # model.load_state_dict(torch.load(f"{model_path}/model/snapshot_fold_99/epoch_{best_epoch}.pth"))
    
    ###################################################################
    # # Experiment 1: reproduce previous results
    # # MSI path
    # train_msi_path = "/home/efs/TrainingData/DFU/BSV+slidingwindow_Processed_V4_centercrop_20240304/cropped_msi_ulcer_only/morph_0" 
    # # unified CSV used in training
    # train_unified_csv = pd.read_csv("/home/efs/ziping/workspaces/dfu/clf_algo_release_202404_rollback/src/data/WAUSI_BSV+shiftwin_20240304.csv")
    # train_unified_csv['GT'] = train_unified_csv['GT_30_old']
    # train_unified_csv.drop(columns = ['GT_30_old', 'GT_30_new'], inplace = True) 
    # train_unified_csv = train_unified_csv[train_unified_csv['good_ori'] == 'Y'].reset_index(drop = True)
    # # unified CSV used in test
    # test_unified_csv = train_unified_csv.copy()
    # # Define test subjects
    # train_subjects = list(train_unified_csv[train_unified_csv['DS_split'] == 'train']['subject_number'].unique())
    # test_set = test_unified_csv[(test_unified_csv['DS_split'] == 'test') & (~test_unified_csv['subject_number'].isin(train_subjects))]['subject_number'].unique().tolist()
    
    # # python3 stratified_analysis.py /home/efs/ziping/workspaces/dfu/clf_algo_release_202404_rollback/src/data/WAUSI_BSV+shiftwin_20240304.csv /home/efs/ziping/workspaces/dfu/clf_algo_release_202404_rollback/results/comparison_study/baseline 571 true
    
    ###################################################################
    # # Experiment 2: sensitivity to change in MSI data
    # # MSI path
    # train_msi_path = "/home/efs/TrainingData/DFU/BSV+slidingwindow_Processed_V4_centercrop_20240422/cropped_msi_ulcer_only/morph_0" 
    # # unified CSV used in training
    # train_unified_csv = pd.read_csv("/home/efs/ziping/workspaces/dfu/clf_algo_release_202404_rollback/src/data/WAUSI_BSV+shiftwin_20240304.csv")
    # train_unified_csv['GT'] = train_unified_csv['GT_30_old']
    # train_unified_csv.drop(columns = ['GT_30_old', 'GT_30_new'], inplace = True) 
    # train_unified_csv = train_unified_csv[train_unified_csv['good_ori'] == 'Y'].reset_index(drop = True)
    # # unified CSV used in test
    # test_unified_csv = train_unified_csv.copy()
    # test_unified_csv = test_unified_csv[test_unified_csv['Visit Number'] == 'DFU_SV1'].reset_index(drop = True)
    # # Define test subjects
    # train_subjects = list(train_unified_csv[train_unified_csv['DS_split'] == 'train']['subject_number'].unique())
    # test_set = test_unified_csv[(test_unified_csv['DS_split'] == 'test') & (~test_unified_csv['subject_number'].isin(train_subjects))]['subject_number'].unique().tolist()

    # # python3 stratified_analysis.py /home/efs/ziping/workspaces/dfu/clf_algo_release_202404_rollback/src/data/WAUSI_BSV+shiftwin_20240304.csv /home/efs/ziping/workspaces/dfu/clf_algo_release_202404_rollback/results/comparison_study/baseline 571 true
    
    ###################################################################
    # # Experiment 3: sensitivity to change in 3D measurements
    # # MSI path
    # train_msi_path = "/home/efs/TrainingData/DFU/BSV+slidingwindow_Processed_V4_centercrop_20240422/cropped_msi_ulcer_only/morph_0" 
    # # unified CSV used in training
    # train_unified_csv = pd.read_csv("/home/efs/ziping/workspaces/dfu/clf_algo_release_202404_rollback/src/data/WAUSI_BSV+shiftwin_20240304.csv")
    # train_unified_csv['GT'] = train_unified_csv['GT_30_old']
    # train_unified_csv.drop(columns = ['GT_30_old', 'GT_30_new'], inplace = True) 
    # train_unified_csv = train_unified_csv[train_unified_csv['good_ori'] == 'Y'].reset_index(drop = True)
    # # unified CSV used in test
    # test_unified_csv = pd.read_csv("/home/efs/ziping/workspaces/dfu/clf_algo_release_202404_rollback/src/data/WAUSI_BSV_oldUnifiedWithNew3DMeasurements_20240417.csv")
    # test_unified_csv['GT'] = test_unified_csv['GT_30_old']
    # test_unified_csv.drop(columns = ['GT_30_old', 'GT_30_new'], inplace = True) 
    # # Define test subjects
    # train_subjects = list(train_unified_csv[train_unified_csv['DS_split'] == 'train']['subject_number'].unique())
    # test_set = test_unified_csv[(test_unified_csv['DS_split'] == 'test') & (~test_unified_csv['subject_number'].isin(train_subjects))]['subject_number'].unique().tolist()

    # # python3 stratified_analysis.py /home/efs/ziping/workspaces/dfu/clf_algo_release_202404_rollback/src/data/WAUSI_BSV_oldUnifiedWithNew3DMeasurements_20240417.csv /home/efs/ziping/workspaces/dfu/clf_algo_release_202404_rollback/results/comparison_study/baseline 571 true

    ###################################################################
    # # Experiment 4: sensitivity to change in GT
    # # MSI path
    # train_msi_path = "/home/efs/TrainingData/DFU/BSV+slidingwindow_Processed_V4_centercrop_20240422/cropped_msi_ulcer_only/morph_0" 
    # # unified CSV used in training
    # train_unified_csv = pd.read_csv("/home/efs/ziping/workspaces/dfu/clf_algo_release_202404_rollback/src/data/WAUSI_BSV+shiftwin_20240304.csv")
    # train_unified_csv['GT'] = train_unified_csv['GT_30_old']
    # train_unified_csv.drop(columns = ['GT_30_old', 'GT_30_new'], inplace = True) 
    # train_unified_csv = train_unified_csv[train_unified_csv['good_ori'] == 'Y'].reset_index(drop = True)
    # # unified CSV used in test
    # test_unified_csv = pd.read_csv("/home/efs/ziping/workspaces/dfu/clf_algo_release_202404_rollback/src/data/WAUSI_BSV_oldUnifiedWithNew3DMeasurements_20240417.csv")
    # test_unified_csv['GT'] = test_unified_csv['GT_30_old']
    # test_unified_csv.drop(columns = ['GT_30_old', 'GT_30_new'], inplace = True)
    # tmp = pd.read_csv("/home/efs/ziping/workspaces/dfu/clf_algo_release_202404_rollback/src/data/WAUSI_unifiedv3_BSVp1-7_reverted_20240418.csv")
    # for idx in range(len(test_unified_csv)):
    #     guid = test_unified_csv.loc[idx, 'ICGUID']
    #     if guid in tmp['ICGUID'].values:
    #         test_unified_csv.loc[idx, 'GT'] = tmp[tmp['ICGUID'] == guid]['GT_30_new'].values[0]
    # test_unified_csv.to_csv("/home/efs/ziping/workspaces/dfu/clf_algo_release_202404_rollback/src/data/WAUSI_BSV_oldUnifiedWithNew3DMeasurementsWithNewGT_20240417.csv")
    
    # # Define test subjects
    # train_subjects = list(train_unified_csv[train_unified_csv['DS_split'] == 'train']['subject_number'].unique())
    # test_set = test_unified_csv[(test_unified_csv['DS_split'] == 'test') & (~test_unified_csv['subject_number'].isin(train_subjects))]['subject_number'].unique().tolist()

    # # python3 stratified_analysis.py /home/efs/ziping/workspaces/dfu/clf_algo_release_202404_rollback/src/data/WAUSI_BSV_oldUnifiedWithNew3DMeasurementsWithNewGT_20240417.csv /home/efs/ziping/workspaces/dfu/clf_algo_release_202404_rollback/results/comparison_study/baseline 571 true

    # ###################################################################
    # # Experiment 5: Performance on unseen new test dataset
    # # MSI path
    # train_msi_path = "/home/efs/TrainingData/DFU/BSV+slidingwindow_Processed_V4_centercrop_20240422/cropped_msi_ulcer_only/morph_0"
    # # unified CSV used in training
    # train_unified_csv = pd.read_csv("/home/efs/ziping/workspaces/dfu/clf_algo_release_202404_rollback/src/data/WAUSI_BSV+shiftwin_20240304.csv")
    # train_unified_csv['GT'] = train_unified_csv['GT_30_old']
    # train_unified_csv.drop(columns = ['GT_30_old', 'GT_30_new'], inplace = True) 
    # train_unified_csv = train_unified_csv[train_unified_csv['good_ori'] == 'Y'].reset_index(drop = True)
    # # unified CSV used in test
    # test_unified_csv = pd.read_csv("/home/efs/ziping/workspaces/dfu/clf_algo_release_202404_rollback/src/data/WAUSI_unifiedv5_BSV+slidingwindow_reverted_newsplit_20240422.csv")
    # test_unified_csv = test_unified_csv[test_unified_csv['good_ori'] == 'Y'].reset_index(drop = True)
    
    # # Define test subjects
    # train_subjects = list(train_unified_csv[train_unified_csv['DS_split'] == 'train']['subject_number'].unique())
    # test_set = test_unified_csv[(~test_unified_csv['subject_number'].isin(train_subjects))]['subject_number'].unique().tolist()
    # # test_set = test_unified_csv[(test_unified_csv['DS_split'] == 'test') & (~test_unified_csv['subject_number'].isin(train_subjects))]['subject_number'].unique().tolist()

    # # python3 stratified_analysis.py /home/efs/ziping/workspaces/dfu/clf_algo_release_202404_rollback/src/data/WAUSI_unifiedv5_BSV+slidingwindow_reverted_newsplit_20240422.csv /home/efs/ziping/workspaces/dfu/clf_algo_release_202404_rollback/results/comparison_study/baseline 571 true
    
    # ###################################################################
    # # Experiment 6: Performance on 5 different randomly shuffled test sets
    # # MSI path
    # train_msi_path = "/home/efs/TrainingData/DFU/BSV+slidingwindow_Processed_V4_centercrop_20240422/cropped_msi_ulcer_only/morph_0"
    # # unified CSV used in training
    # train_unified_csv = pd.read_csv("/home/efs/ziping/workspaces/dfu/clf_algo_release_202404_rollback/src/data/WAUSI_BSV+shiftwin_20240304.csv")
    # train_unified_csv['GT'] = train_unified_csv['GT_30_old']
    # train_unified_csv.drop(columns = ['GT_30_old', 'GT_30_new'], inplace = True) 
    # train_unified_csv = train_unified_csv[train_unified_csv['good_ori'] == 'Y'].reset_index(drop = True)
    # # unified CSV used in test
    # # test_unified_csv = pd.read_csv("/home/efs/ziping/workspaces/dfu/clf_algo_release_202404_rollback/src/data/WAUSI_unifiedv6_BSV+slidingwindow_reverted_randsplit_1_20240430.csv")
    # # test_unified_csv = pd.read_csv("/home/efs/ziping/workspaces/dfu/clf_algo_release_202404_rollback/src/data/WAUSI_unifiedv6_BSV+slidingwindow_reverted_randsplit_2_20240430.csv")
    # # test_unified_csv = pd.read_csv("/home/efs/ziping/workspaces/dfu/clf_algo_release_202404_rollback/src/data/WAUSI_unifiedv6_BSV+slidingwindow_reverted_randsplit_3_20240430.csv")
    # # test_unified_csv = pd.read_csv("/home/efs/ziping/workspaces/dfu/clf_algo_release_202404_rollback/src/data/WAUSI_unifiedv6_BSV+slidingwindow_reverted_randsplit_4_20240430.csv")
    # test_unified_csv = pd.read_csv("/home/efs/ziping/workspaces/dfu/clf_algo_release_202404_rollback/src/data/WAUSI_unifiedv6_BSV+slidingwindow_reverted_randsplit_5_20240430.csv")
    # test_unified_csv = test_unified_csv[test_unified_csv['good_ori'] == 'Y'].reset_index(drop = True)
    
    # # Define test subjects
    # train_subjects = list(train_unified_csv[train_unified_csv['DS_split'] == 'train']['subject_number'].unique())
    # test_set = test_unified_csv[(~test_unified_csv['subject_number'].isin(train_subjects))]['subject_number'].unique().tolist()
    # # test_set = test_unified_csv[(test_unified_csv['DS_split'] == 'test') & (~test_unified_csv['subject_number'].isin(train_subjects))]['subject_number'].unique().tolist()
    
    # # Export path
    # save_path = "/home/efs/ziping/workspaces/dfu/clf_algo_release_202404_rollback/results/robustness_study_old_shiftwin/baseline/hs_571/predictions_test.csv"

    # # python3 stratified_analysis.py ./data/WAUSI_unifiedv6_BSV+slidingwindow_reverted_randsplit_1_20240430.csv /home/efs/ziping/workspaces/dfu/clf_algo_release_202404_rollback/results/robustness_study_old_shiftwin/baseline 571 true true    
    # # python3 stratified_analysis.py ./data/WAUSI_unifiedv6_BSV+slidingwindow_reverted_randsplit_2_20240430.csv /home/efs/ziping/workspaces/dfu/clf_algo_release_202404_rollback/results/robustness_study_old_shiftwin/baseline 571 true true    
    # # python3 stratified_analysis.py ./data/WAUSI_unifiedv6_BSV+slidingwindow_reverted_randsplit_3_20240430.csv /home/efs/ziping/workspaces/dfu/clf_algo_release_202404_rollback/results/robustness_study_old_shiftwin/baseline 571 true true    
    # # python3 stratified_analysis.py ./data/WAUSI_unifiedv6_BSV+slidingwindow_reverted_randsplit_4_20240430.csv /home/efs/ziping/workspaces/dfu/clf_algo_release_202404_rollback/results/robustness_study_old_shiftwin/baseline 571 true true    
    # # python3 stratified_analysis.py ./data/WAUSI_unifiedv6_BSV+slidingwindow_reverted_randsplit_5_20240430.csv /home/efs/ziping/workspaces/dfu/clf_algo_release_202404_rollback/results/robustness_study_old_shiftwin/baseline 571 true true    
    
    ###################################################################
    # Experiment 7: Performance on 5 different randomly shuffled test sets
    # MSI path
    train_msi_path = "/home/efs/TrainingData/DFU/BSV+slidingwindow_Processed_V4_centercrop_20240422/cropped_msi_ulcer_only/morph_0"
    # unified CSV used in training
    train_unified_csv = pd.read_csv("/home/efs/ziping/workspaces/dfu/clf_algo_release_202404_rollback/src/data/WAUSI_BSV+shiftwin_20240304.csv")
    train_unified_csv['GT'] = train_unified_csv['GT_30_old']
    train_unified_csv.drop(columns = ['GT_30_old', 'GT_30_new'], inplace = True) 
    train_unified_csv = train_unified_csv[train_unified_csv['good_ori'] == 'Y'].reset_index(drop = True)
    # unified CSV used in test
    # test_unified_csv = pd.read_csv("/home/efs/ziping/workspaces/dfu/clf_algo_release_202404_rollback/src/data/WAUSI_unifiedv6_BSV+slidingwindow_reverted_randsplit_5_20240430.csv")
    test_unified_csv = pd.read_csv("/home/efs/ziping/workspaces/dfu/clf_algo_release_202404_rollback/src/data/WAUSI_unifiedv7_BSV+slidingwindow_reverted_final_test_20240501.csv")
    
    # Define test subjects
    train_subjects = list(train_unified_csv[train_unified_csv['DS_split'] == 'train']['subject_number'].unique())
    # test_set = ['203-088', '203-091', '202-049', '202-042', '202-037', '205-001', '203-083', '201-032', '201-018', '202-043', '201-009', '202-070', '201-022', '202-073', 
    #             '202-039', '202-052', '201-028', '203-086', '203-087', '203-046', '202-055', '202-020', '203-065', '202-080', '202-082', '202-053', '202-051', '202-041', 
    #             '203-090', '202-083', '203-053', '202-075', '203-085', '202-011', '202-072', '203-069', '203-067', '202-048', '202-010', '201-023']
    
    # test_set = ['201-029', '202-080', '203-070', '201-035', '205-002', '202-051', '203-091', '202-009', '203-085', '203-089', '202-057', '203-052', '203-065', '202-063', 
    #             '202-073', '202-011', '203-030', '201-033', '203-086', '203-074', '202-047', '203-072', '202-083', '202-043', '201-031', '202-078', '203-084', '202-042', 
    #             '202-039', '202-062', '202-052', '202-022', '201-024', '202-064', '201-028', '202-072', '202-050', '205-001', '203-088', '203-083']

    # test_set = ['202-040', '201-031', '202-039', '202-075', '202-069', '203-014', '202-010', '202-064', '202-050', '203-065', '202-076', '202-020', '201-028', '202-067', 
    #             '203-090', '202-078', '202-044', '202-082', '203-069', '203-074', '202-047', '202-055', '202-083', '205-002', '203-030', '202-037', '203-089', '202-063', 
    #             '203-046', '202-011', '203-053', '202-042', '202-022', '202-054', '202-072', '201-022', '202-041', '202-009', '201-035', '202-068']

    # test_set = ['202-047', '202-056', '203-082', '202-010', '202-041', '202-069', '202-051', '203-090', '203-069', '203-084', '202-052', '201-024', '202-076', '202-062', 
    #             '203-065', '202-040', '203-086', '201-028', '205-002', '202-080', '202-050', '203-067', '201-033', '203-091', '203-085', '201-018', '202-067', '201-009', 
    #             '201-019', '202-068', '201-029', '203-014', '202-039', '202-044', '203-083', '202-072', '202-049', '202-054', '201-034', '202-022']

    # test_set = ['202-010', '202-078', '202-052', '202-049', '201-022', '203-030', '201-018', '202-044', '202-011', '202-053', '202-073', '202-070', '205-003', '202-067', 
    #             '202-022', '202-069', '202-056', '202-064', '203-086', '202-041', '202-068', '203-072', '201-019', '202-042', '202-082', '203-065', '203-087', '201-028', 
    #             '203-053', '201-031', '202-080', '203-085', '203-070', '202-048', '203-014', '201-034', '202-039', '202-043', '205-002', '202-075']

    # test_set = ['202-037', '202-009', '203-083', '202-046', '203-084', '203-090', '203-086', '201-032', '202-041', '203-069', '202-039', '202-048', '202-082', '201-034', 
    #             '202-067', '205-002', '203-070', '201-030', '203-030', '202-051', '203-053', '202-050', '202-069', '202-068', '201-024', '201-035', '203-091', '202-083', 
    #             '203-052', '202-080', '202-076', '202-020', '202-057', '203-088', '203-067', '202-054', '202-070', '203-087', '202-044', '202-062']
    
    test_set = test_unified_csv[(~test_unified_csv['subject_number'].isin(train_subjects))]['subject_number'].unique().tolist()
        
    # Export path
    save_path = "/home/efs/ziping/workspaces/dfu/clf_algo_release_202404_rollback/results/old_shiftwin_generalization_results/baseline/hs_571/predictions_test.csv"
    
    # python3 stratified_analysis.py ./data/WAUSI_unifiedv6_BSV+slidingwindow_reverted_randsplit_5_20240430.csv /home/efs/ziping/workspaces/dfu/clf_algo_release_202404_rollback/results/final_test_old_shiftwin/baseline 571 true true

    # ###################################################################
    # # Experiment 8: Performance on final test set
    # # MSI path
    # train_msi_path = "/home/efs/TrainingData/DFU/BSV+slidingwindow_Processed_V4_centercrop_20240422/cropped_msi_ulcer_only/morph_0"
    # # unified CSV used in training
    # train_unified_csv = pd.read_csv("/home/efs/ziping/workspaces/dfu/clf_algo_release_202404_rollback/src/data/WAUSI_BSV+shiftwin_20240304.csv")
    # train_unified_csv['GT'] = train_unified_csv['GT_30_old']
    # train_unified_csv.drop(columns = ['GT_30_old', 'GT_30_new'], inplace = True) 
    # train_unified_csv = train_unified_csv[train_unified_csv['good_ori'] == 'Y'].reset_index(drop = True)
    # # unified CSV used in test
    # test_unified_csv = pd.read_csv("/home/efs/ziping/workspaces/dfu/clf_algo_release_202404_rollback/src/data/WAUSI_unifiedv7_BSV+slidingwindow_reverted_final_test_20240501.csv")
    
    # # Define test subjects
    # train_subjects = list(train_unified_csv[train_unified_csv['DS_split'] == 'train']['subject_number'].unique())
    # test_set = test_unified_csv[(~test_unified_csv['subject_number'].isin(train_subjects))]['subject_number'].unique().tolist()
    
    # # Export path
    # save_path = "/home/efs/ziping/workspaces/dfu/clf_algo_release_202404_rollback/results/final_test_old_shiftwin/baseline/hs_571/predictions_test.csv"
    
    # # python3 stratified_analysis.py ./data/WAUSI_unifiedv7_BSV+slidingwindow_reverted_final_test_20240501.csv /home/efs/ziping/workspaces/dfu/clf_algo_release_202404_rollback/results/final_test_old_shiftwin/baseline 571 true true
    
    # ###################################################################
    # # Experiment 9: Performance on final test set
    # # MSI path
    # train_msi_path = "/home/efs/TrainingData/DFU/BSV+slidingwindow_Processed_V4_centercrop_20240422/cropped_msi_ulcer_only/morph_0"
    # # unified CSV used in training
    # train_unified_csv = pd.read_csv("/home/efs/ziping/workspaces/dfu/clf_algo_release_202404_rollback/src/data/WAUSI_unifiedv7_BSV+slidingwindow_reverted_randsplit_5_20240501.csv")
    # train_unified_csv = train_unified_csv[train_unified_csv['good_ori'] == 'Y'].reset_index(drop = True)
    # # unified CSV used in test
    # # test_unified_csv = pd.read_csv("/home/efs/ziping/workspaces/dfu/clf_algo_release_202404_rollback/src/data/WAUSI_unifiedv7_BSV+slidingwindow_reverted_final_test_20240501.csv")
    # test_unified_csv = pd.read_csv("/home/efs/ziping/workspaces/dfu/clf_algo_release_202404_rollback/src/data/WAUSI_unifiedv7_BSV+slidingwindow_reverted_randsplit_5_20240501.csv")
    # test_unified_csv = test_unified_csv[test_unified_csv['good_ori'] == 'Y'].reset_index(drop = True)
    
    # # Define test subjects
    # train_subjects = list(train_unified_csv[train_unified_csv['DS_split'] == 'train']['subject_number'].unique())
    # test_set = test_unified_csv[(~test_unified_csv['subject_number'].isin(train_subjects))]['subject_number'].unique().tolist()
    
    # # Export path
    # save_path = "/home/efs/ziping/workspaces/dfu/clf_algo_release_202404_rollback/results/final_test_new_shiftwin/baseline/hs_571/predictions_test.csv"
    
    # # python3 stratified_analysis.py ./data/WAUSI_unifiedv7_BSV+slidingwindow_reverted_final_test_20240501.csv ../results/final_test_new_shiftwin/baseline 994 true true

    assert(len(set(test_set).intersection(set(train_subjects))) == 0)
    
    # Train/Test split
    df_train = train_unified_csv[train_unified_csv['DS_split'] == 'train'].reset_index(drop = True)
    df_test = test_unified_csv[test_unified_csv['subject_number'].isin(test_set)].reset_index(drop = True)
    
    # Preprocess clinical features
    preprocess = PreProcessing(config, df_train, df_test)
    X_train_feat, X_test_feat = preprocess.preprocess_df()
    assert sum(sum(X_train_feat.values - pd.read_csv(f"{model_path}/X_train_feat_fold99.csv").values)) < 1e-3
            
    X_test_img = pd.DataFrame({
                'subject_number': df_test['subject_number'],
                'Visit Number': df_test['Visit Number'],
                'ICGUID': df_test['ICGUID'],
                'image': train_msi_path + "/data/" + df_test['subject_number'] + "-" + df_test['ICGUID'] + ".npy",
                'mask': train_msi_path + "/mask/" + df_test['subject_number'] + "-" + df_test['ICGUID'] + "_mask.npy",
                })
    y_test = df_test[['GT']]
            
    test_dataset = DFUDataset(config,
                            X_test_feat,
                            X_test_img,
                            y_test,
                            indicator = 'test')
    test_loader = DataLoader(test_dataset,
                            batch_size = int(config.BATCH_SIZE),
                            shuffle = False,
                            num_workers = config.NUM_WORKERS)
            
    res_df = predict(test_loader, model, 0)
           
    res_df.to_csv(save_path, index = False)