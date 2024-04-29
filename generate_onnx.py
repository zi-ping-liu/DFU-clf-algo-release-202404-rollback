# Import libraries
import os, sys
sys.path.append("/home/efs/ziping/workspaces/dfu/clf_algo_release_202404_rollback/src/")
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from models.baseline_onnx_fix import BaseLine
from utils.dfu_dataset import DFUDataset
from torch.utils.data import DataLoader
from utils.preprocess import PreProcessing
import onnxruntime as ort



class Config:
    def __init__(self, data):
        for key, value in data.items():
            setattr(self, key, value)


# Onnx export path
onnx_export_path = "/home/efs/ziping/workspaces/dfu/clf_algo_release_202404_rollback/results/onnx_model/baseline_shitwin_240426.onnx"

model_path = "/home/efs/ziping/workspaces/dfu/clf_algo_release_202404_rollback/results/reproduce_shiftwin/baseline/hs_571"
best_epoch = 14
config = np.load(f"{model_path}/model/config_CV.npy", allow_pickle = True).item()
config = Config(config)
device = 0
    
if (not os.path.exists(onnx_export_path)): # Generate Onnx model
    
    # Load model to be converted
    model = BaseLine(config).to(device)
    model.load_state_dict(torch.load(f"{model_path}/model/snapshot_fold_99/epoch_{best_epoch}.pth"))
    for param in model.parameters():
        param.data = param.data.detach()  # Replace the parameter with its detached version
    model.eval()

    # Convert to onnx model
    clin_features = torch.rand(1, 47).to(torch.float32).to(device)
    images = torch.rand(1, 8, 700, 700).to(device)
    mask = torch.rand(1, 1, 700, 700).to(device)
    with torch.no_grad():
        #creating onnx model
        torch.onnx.export(model, # model being run
                        (clin_features, images, mask), # model input (or a tuple for multiple inputs)
                        onnx_export_path, # export path
                        export_params = True, # store the trained parameter weights inside the model file
                        opset_version = 12, # the ONNX version to export the model to
                        input_names = ['clinical', 'image', 'mask'], # the model's input names
                        output_names = ['output'], # the model's output names
                        verbose = True,
                        dynamic_axes = {
                            'clinical' : {0 : 'batch_size'},
                            'image': {0 : 'batch_size'},
                            'mask': {0 : 'batch_size'},
                            'output': {0 : 'batch_size'}
                            })
        
else: # Perform inference
    
    ###################################################################
    # Experiment 1: reproduce previous results
    # MSI path
    train_msi_path = "/home/efs/TrainingData/DFU/BSV+slidingwindow_Processed_V4_centercrop_20240304/cropped_msi_ulcer_only/morph_0" 
    # unified CSV used in training
    train_unified_csv = pd.read_csv("/home/efs/ziping/workspaces/dfu/clf_algo_release_202404_rollback/src/data/WAUSI_BSV+shiftwin_20240304.csv")
    train_unified_csv['GT'] = train_unified_csv['GT_30_old']
    train_unified_csv.drop(columns = ['GT_30_old', 'GT_30_new'], inplace = True) 
    train_unified_csv = train_unified_csv[train_unified_csv['good_ori'] == 'Y'].reset_index(drop = True)
    # unified CSV used in test
    test_unified_csv = train_unified_csv.copy()
    # Define test subjects
    train_subjects = list(train_unified_csv[train_unified_csv['DS_split'] == 'train']['subject_number'].unique())
    test_set = test_unified_csv[(test_unified_csv['DS_split'] == 'test') & (~test_unified_csv['subject_number'].isin(train_subjects))]['subject_number'].unique().tolist()
    
    pred_export_path = "/home/efs/ziping/workspaces/dfu/clf_algo_release_202404_rollback/results/onnx_check/baseline/hs_571/predictions_test.csv"
    # python3 stratified_analysis.py /home/efs/ziping/workspaces/dfu/clf_algo_release_202404_rollback/src/data/WAUSI_BSV+shiftwin_20240304.csv /home/efs/ziping/workspaces/dfu/clf_algo_release_202404_rollback/results/onnx_check/baseline 571 true

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
    
    # pred_export_path = "/home/efs/ziping/workspaces/dfu/clf_algo_release_202404_rollback/results/onnx_check/baseline/hs_571/predictions_test.csv"
    # # python3 stratified_analysis.py /home/efs/ziping/workspaces/dfu/clf_algo_release_202404_rollback/src/data/WAUSI_unifiedv5_BSV+slidingwindow_reverted_newsplit_20240422.csv /home/efs/ziping/workspaces/dfu/clf_algo_release_202404_rollback/results/onnx_check/baseline 571 true


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
                            batch_size = 1,
                            shuffle = False,
                            num_workers = 1)

    # Load the ONNX model
    session = ort.InferenceSession(onnx_export_path)

    res_df = pd.DataFrame()

    for _, (clin_features, images, mask, target, (subject_number, visit_number, icguid)) in enumerate(test_loader, start = 1):
        
        clin_features = clin_features.to(torch.float32).to(device)     
        images = images.to(device) # shape: (batch_size, C, H, W)
        mask = mask.to(device)
        
        target = torch.squeeze(target).long()
        if target.ndim == 0: target = target.unsqueeze(0)

        inputs = {session.get_inputs()[0].name: clin_features.detach().cpu().numpy(),
                session.get_inputs()[1].name: images.detach().cpu().numpy(),
                session.get_inputs()[2].name: mask.detach().cpu().numpy()}
        outputs = np.array(session.run(None, inputs)).reshape(1,2)
        pred = (np.exp(outputs) / np.sum(np.exp(outputs)))[0][1]
        
        cur_df = pd.DataFrame({
                    'subject_number': list(subject_number),
                    'Visit Number': list(visit_number),
                    'ICGUID': list(icguid),
                    'Pred_Proba': pred,
                    'GT': target.cpu().numpy()
                })
        res_df = pd.concat([res_df, cur_df], ignore_index = True)
        
    res_df.to_csv(pred_export_path, index = False)