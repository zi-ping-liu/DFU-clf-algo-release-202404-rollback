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
    
    train_csv = sys.argv[1]
    test_csv = sys.argv[2]
    test_set = int(sys.argv[3])
    model_path = sys.argv[4]
    best_epoch = int(sys.argv[5])
    save_path = sys.argv[6]
    
    # Export path
    export_path = '/'.join(save_path.split('/')[:-1])
    if not os.path.exists(export_path): os.makedirs(export_path)
    
    # Load model weights
    config = np.load(f"{model_path}/model/config_final_train.npy", allow_pickle = True).item()
    config = Config(config)
    model = BaseLine(config).to(0)
    model.load_state_dict(torch.load(f"{model_path}/model/snapshot_fold_99/epoch_{best_epoch}.pth"))
    
    # MSI path
    train_msi_path = "/home/efs/TrainingData/DFU/BSV+slidingwindow_Processed_V4_centercrop_20240422/cropped_msi_ulcer_only/morph_0"
    # unified CSV used in training
    train_csv = pd.read_csv(train_csv)
    train_csv = train_csv[train_csv['good_ori'] == 'Y'].reset_index(drop = True)
    # unified CSV used in test
    test_csv = pd.read_csv(test_csv)
    
    # Define test subjects
    train_subjects = list(train_csv[train_csv['DS_split'] == 'train']['subject_number'].unique())
    
    if test_set == 0:
        test_subjects = test_csv[(~test_csv['subject_number'].isin(train_subjects))]['subject_number'].unique().tolist()
    elif test_set == 1:
        test_subjects = ['201-029', '202-080', '203-070', '201-035', '205-002', '202-051', '203-091', '202-009', '203-085', '203-089', '202-057', '203-052', '203-065', '202-063', 
                         '202-073', '202-011', '203-030', '201-033', '203-086', '203-074', '202-047', '203-072', '202-083', '202-043', '201-031', '202-078', '203-084', '202-042', 
                         '202-039', '202-062', '202-052', '202-022', '201-024', '202-064', '201-028', '202-072', '202-050', '205-001', '203-088', '203-083']
    elif test_set == 2:
        test_subjects = ['202-040', '201-031', '202-039', '202-075', '202-069', '203-014', '202-010', '202-064', '202-050', '203-065', '202-076', '202-020', '201-028', '202-067', 
                         '203-090', '202-078', '202-044', '202-082', '203-069', '203-074', '202-047', '202-055', '202-083', '205-002', '203-030', '202-037', '203-089', '202-063', 
                         '203-046', '202-011', '203-053', '202-042', '202-022', '202-054', '202-072', '201-022', '202-041', '202-009', '201-035', '202-068']
    elif test_set == 3:
        test_subjects = ['202-047', '202-056', '203-082', '202-010', '202-041', '202-069', '202-051', '203-090', '203-069', '203-084', '202-052', '201-024', '202-076', '202-062', 
                         '203-065', '202-040', '203-086', '201-028', '205-002', '202-080', '202-050', '203-067', '201-033', '203-091', '203-085', '201-018', '202-067', '201-009', 
                         '201-019', '202-068', '201-029', '203-014', '202-039', '202-044', '203-083', '202-072', '202-049', '202-054', '201-034', '202-022']
    elif test_set == 4:
        test_subjects = ['202-010', '202-078', '202-052', '202-049', '201-022', '203-030', '201-018', '202-044', '202-011', '202-053', '202-073', '202-070', '205-003', '202-067', 
                         '202-022', '202-069', '202-056', '202-064', '203-086', '202-041', '202-068', '203-072', '201-019', '202-042', '202-082', '203-065', '203-087', '201-028', 
                         '203-053', '201-031', '202-080', '203-085', '203-070', '202-048', '203-014', '201-034', '202-039', '202-043', '205-002', '202-075']
    elif test_set == 5:
        test_subjects = ['202-037', '202-009', '203-083', '202-046', '203-084', '203-090', '203-086', '201-032', '202-041', '203-069', '202-039', '202-048', '202-082', '201-034', 
                         '202-067', '205-002', '203-070', '201-030', '203-030', '202-051', '203-053', '202-050', '202-069', '202-068', '201-024', '201-035', '203-091', '202-083', 
                         '203-052', '202-080', '202-076', '202-020', '202-057', '203-088', '203-067', '202-054', '202-070', '203-087', '202-044', '202-062']
    else:
        raise Exception("'test_set' out of range: select between 0 and 5.")
    assert(len(set(test_subjects).intersection(set(train_subjects))) == 0)
    
    # Train/Test split
    df_train = train_csv[train_csv['subject_number'].isin(train_subjects)].reset_index(drop = True)
    df_test = test_csv[test_csv['subject_number'].isin(test_subjects)].reset_index(drop = True)
    
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