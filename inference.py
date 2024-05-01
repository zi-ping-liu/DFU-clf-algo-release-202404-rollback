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
    
    # Load model weights
    model_path = "/home/efs/ziping/workspaces/dfu/clf_algo_release_202404_rollback/results/reproduce_shiftwin/baseline/hs_571"
    best_epoch = 14
    config = np.load(f"{model_path}/model/config_CV.npy", allow_pickle = True).item()
    config = Config(config)
    model = BaseLine(config).to(0)
    model.load_state_dict(torch.load(f"{model_path}/model/snapshot_fold_99/epoch_{best_epoch}.pth"))
    
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
    
    ###################################################################
    # Experiment 6: Performance on 5 different randomly shuffled test sets
    # MSI path
    train_msi_path = "/home/efs/TrainingData/DFU/BSV+slidingwindow_Processed_V4_centercrop_20240422/cropped_msi_ulcer_only/morph_0"
    # unified CSV used in training
    train_unified_csv = pd.read_csv("/home/efs/ziping/workspaces/dfu/clf_algo_release_202404_rollback/src/data/WAUSI_BSV+shiftwin_20240304.csv")
    train_unified_csv['GT'] = train_unified_csv['GT_30_old']
    train_unified_csv.drop(columns = ['GT_30_old', 'GT_30_new'], inplace = True) 
    train_unified_csv = train_unified_csv[train_unified_csv['good_ori'] == 'Y'].reset_index(drop = True)
    # unified CSV used in test
    # test_unified_csv = pd.read_csv("/home/efs/ziping/workspaces/dfu/clf_algo_release_202404_rollback/src/data/WAUSI_unifiedv6_BSV+slidingwindow_reverted_randsplit_1_20240430.csv")
    # test_unified_csv = pd.read_csv("/home/efs/ziping/workspaces/dfu/clf_algo_release_202404_rollback/src/data/WAUSI_unifiedv6_BSV+slidingwindow_reverted_randsplit_2_20240430.csv")
    # test_unified_csv = pd.read_csv("/home/efs/ziping/workspaces/dfu/clf_algo_release_202404_rollback/src/data/WAUSI_unifiedv6_BSV+slidingwindow_reverted_randsplit_3_20240430.csv")
    # test_unified_csv = pd.read_csv("/home/efs/ziping/workspaces/dfu/clf_algo_release_202404_rollback/src/data/WAUSI_unifiedv6_BSV+slidingwindow_reverted_randsplit_4_20240430.csv")
    test_unified_csv = pd.read_csv("/home/efs/ziping/workspaces/dfu/clf_algo_release_202404_rollback/src/data/WAUSI_unifiedv6_BSV+slidingwindow_reverted_randsplit_5_20240430.csv")
    test_unified_csv = test_unified_csv[test_unified_csv['good_ori'] == 'Y'].reset_index(drop = True)
    
    # Define test subjects
    train_subjects = list(train_unified_csv[train_unified_csv['DS_split'] == 'train']['subject_number'].unique())
    test_set = test_unified_csv[(~test_unified_csv['subject_number'].isin(train_subjects))]['subject_number'].unique().tolist()
    # test_set = test_unified_csv[(test_unified_csv['DS_split'] == 'test') & (~test_unified_csv['subject_number'].isin(train_subjects))]['subject_number'].unique().tolist()
    
    # Export path
    save_path = "/home/efs/ziping/workspaces/dfu/clf_algo_release_202404_rollback/results/robustness_study_old_shiftwin/baseline/hs_571/predictions_test.csv"

    # python3 stratified_analysis.py ./data/WAUSI_unifiedv6_BSV+slidingwindow_reverted_randsplit_1_20240430.csv /home/efs/ziping/workspaces/dfu/clf_algo_release_202404_rollback/results/robustness_study_old_shiftwin/baseline 571 true true    
    # python3 stratified_analysis.py ./data/WAUSI_unifiedv6_BSV+slidingwindow_reverted_randsplit_2_20240430.csv /home/efs/ziping/workspaces/dfu/clf_algo_release_202404_rollback/results/robustness_study_old_shiftwin/baseline 571 true true    
    # python3 stratified_analysis.py ./data/WAUSI_unifiedv6_BSV+slidingwindow_reverted_randsplit_3_20240430.csv /home/efs/ziping/workspaces/dfu/clf_algo_release_202404_rollback/results/robustness_study_old_shiftwin/baseline 571 true true    
    # python3 stratified_analysis.py ./data/WAUSI_unifiedv6_BSV+slidingwindow_reverted_randsplit_4_20240430.csv /home/efs/ziping/workspaces/dfu/clf_algo_release_202404_rollback/results/robustness_study_old_shiftwin/baseline 571 true true    
    # python3 stratified_analysis.py ./data/WAUSI_unifiedv6_BSV+slidingwindow_reverted_randsplit_5_20240430.csv /home/efs/ziping/workspaces/dfu/clf_algo_release_202404_rollback/results/robustness_study_old_shiftwin/baseline 571 true true    
    
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