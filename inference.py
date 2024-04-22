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
    
    # Export path
    save_path = "/home/efs/ziping/workspaces/dfu/clf_algo_release_202404_rollback/results/comparison_study/baseline/hs_571/predictions_test.csv"
    
    # MSI path
    # train_msi_path = "/home/efs/TrainingData/DFU/BSV_203subj_Processed_V4_centercrop_20240411/cropped_msi_ulcer_only/morph_0"
    train_msi_path = "/home/efs/TrainingData/DFU/BSV+slidingwindow_Processed_V4_centercrop_20240422/cropped_msi_ulcer_only/morph_0"
    
    # Load unified CSV used in early validation
    test_unified_csv = pd.read_csv("/home/efs/ziping/workspaces/dfu/clf_algo_release_202404_rollback/src/data/WAUSI_unifiedv4_BSV_reverted_20240422.csv")
    test_unified_csv = test_unified_csv[test_unified_csv['good_ori'] == 'Y'].reset_index(drop = True)
    
    # Get new test subject list (remove overlap)
    df_old = pd.read_csv("/home/efs/ziping/workspaces/dfu/clf_algo_release_202404_rollback/src/data/processed_WAUSI_BSV+shiftwin_20240304.csv")
    df_old = df_old[df_old['good_ori'] == 'Y'].reset_index(drop = True)
    train_subjects = list(df_old[df_old['DS_split'] == 'train']['subject_number'].unique())
    df_new = pd.read_csv("/home/efs/ziping/workspaces/dfu/clf_algo_release_202404_rollback/src/data/WAUSI_unifiedv4_BSV_reverted_20240422.csv")
    # test_set = df_new[(df_new['DS_split'] == 'test') & (~df_new['subject_number'].isin(train_subjects))]['subject_number'].unique().tolist()
    test_set = df_new[(~df_new['DS_split'].isin(['bad_quality', 'exclude_from_classification'])) & (~df_new['subject_number'].isin(train_subjects))]['subject_number'].unique().tolist()
    #######################################################################################################################
    

    # Train/Test split
    df = pd.read_csv("/home/efs/ziping/workspaces/dfu/clf_algo_release_202404_rollback/src/data/processed_WAUSI_BSV+shiftwin_20240304.csv")
    df = df[df['good_ori'] == 'Y'].reset_index(drop = True)    
    df_train = df[df['DS_split'] == 'train'].reset_index(drop = True)
    df_test = test_unified_csv[test_unified_csv['subject_number'].isin(test_set)].reset_index(drop = True)
              
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