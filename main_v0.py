"""
Main source codes: Model training with K-fold cross-validation (optional)

Author: Ziping Liu
Date: Apr 19, 2024
"""



# Import libraries
import os
import random
import time
import copy
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from utils.dfu_dataset import DFUDataset
from utils.preprocess import PreProcessing
from models.baseline import BaseLine
from utils.metrics_monitor import MetricMonitor
from utils.clf_loss import weighted_CE
from utils import clf_metrics



def main(config, seed = 1337):
    
    torch.cuda.set_device(config.DEVICE)
    
    # Avoid previous training history from being overwritten
    if (config.K_FOLD == 1) and (os.path.exists(f"{config.EXPORT_PATH}/{config.VERSION}/model/config_final_train.npy")):
        raise Exception(f"Terminated. Final train already completed for this hyperparameter set.\n")
    
    if (config.K_FOLD != 1) and (os.path.exists(f"{config.EXPORT_PATH}/{config.VERSION}/model/config_CV.npy")):
        raise Exception(f"Terminated. K-fold CV already completed for this hyperparameter set.\n")
    
    # Ensure experiment reproducibility
    set_seed(seed)
    
    print("==============================================================================")
    print("                              Training Info                                   ")
    df = pd.read_csv(config.TRAIN_CSV) # Load WAUSI unified CSV
    print(f"> Unified CSV for WAUSI study loaded ({df.shape[0]} rows, {df.shape[1]} columns)")
    
    if (config.IMAGE_PREPROCESS['USE_GOOD_ORI_ONLY']): # Keep only good-orientation cases
        df = df[df['good_ori'] == 'Y'].reset_index(drop = True)
        print(f"    >> Bad-orientation rows dropped ({df.shape[0]} rows remaining)")
    
    # Separate data into training/test sets
    df_train = df[df['subject_number'].str.strip().isin(config.TRAIN_SET)].reset_index(drop = True)
    df_test = df[df['subject_number'].str.strip().isin(config.TEST_SET)].reset_index(drop = True)
    
    config.NUM_TRAIN_SUBJECTS = len(df_train['subject_number'].unique())
    config.NUM_TRAIN_IMAGES = len(df_train)
    
    config.NUM_TEST_SUBJECTS = len(df_test['subject_number'].unique())
    config.NUM_TEST_IMAGES = len(df_test)
    
    print("\n> Train/test split:")
    print(f"    >> Training Set: {config.NUM_TRAIN_SUBJECTS} subjects, {config.NUM_TRAIN_IMAGES} ICGUIDs")
    print(f"    >> Test Set: {config.NUM_TEST_SUBJECTS} subjects, {config.NUM_TEST_IMAGES} ICGUIDs")
    
    # K-fold-split the training dataset
    df_train['fold'] = 99
    df_test['fold'] = 99
    if (config.K_FOLD > 1): # Assign subjects manually
        # Fold 1
        fold_1_ids = ['201-005', '201-008', '201-012', '201-017', '202-002', '202-028', '202-030', '203-005', 
                      '203-010', '203-034', '203-036', '203-041', '203-048', '203-051', '203-060']
        assign_1 = df_train['subject_number'].isin(fold_1_ids)
        df_train.loc[assign_1, 'fold'] = 1

        # Fold 2
        fold_2_ids = ['201-007', '201-016', '201-025', '202-004', '202-017', '202-023', '202-024', '203-004',
                      '203-013', '203-015', '203-024', '203-025', '203-037', '203-047', '203-054']
        assign_2 = df_train['subject_number'].isin(fold_2_ids)
        df_train.loc[assign_2, 'fold'] = 2
        
        # Fold 3
        fold_3_ids = ['201-001', '201-006', '201-010', '201-011', '201-015', '202-013', '202-016', '202-029', 
                      '202-035', '203-012', '203-020', '203-032', '203-038', '203-058', '203-061']
        assign_3 = df_train['subject_number'].isin(fold_3_ids)
        df_train.loc[assign_3, 'fold'] = 3
        
        # Fold 4
        fold_4_ids = ['201-003', '201-014', '202-012', '202-015', '202-018', '202-021', '202-031', '203-002', 
                      '203-011', '203-017', '203-021', '203-028', '203-039', '203-044', '203-050']
        assign_4 = df_train['subject_number'].isin(fold_4_ids)
        df_train.loc[assign_4, 'fold'] = 4
        
        # Fold 5
        fold_5_ids = ['201-013', '202-008', '202-014', '202-019', '202-025', '202-032', '203-001', '203-006', 
                      '203-018', '203-029', '203-031', '203-033', '203-043', '203-045', '203-057', '203-066']
        assign_5 = df_train['subject_number'].isin(fold_5_ids)
        df_train.loc[assign_5, 'fold'] = 5
    
    assert len(df_train['fold'].unique()) == config.K_FOLD # Ensure the correctness of K-fold assignment
    
    print_exp_info(config)

    print(f"\n******************************************************************************")
    if (config.K_FOLD == 1):
        print("Performing final train")
        config.FINAL_TRAIN = True
    else:
        print(f"Performing {config.K_FOLD}-fold cross validation")
        config.FINAL_TRAIN = False
    print("******************************************************************************")
    
    config.MODEL_SAVE_PATH = config.EXPORT_PATH + '/' + config.VERSION + "/model"
    if not os.path.exists(config.MODEL_SAVE_PATH): os.makedirs(config.MODEL_SAVE_PATH)
            
    config.PREDICTION_PATH = config.EXPORT_PATH + '/' + config.VERSION + "/predictions"
    if not os.path.exists(config.PREDICTION_PATH): os.makedirs(config.PREDICTION_PATH)
    
    if config.FINAL_TRAIN:
        
        fold = 99
        
        preprocess = PreProcessing(config, df_train, df_test)
        X_train_feat, X_test_feat = preprocess.preprocess_df()
        
        config.NUM_CLINICAL_FEATURES = X_train_feat.shape[1]
        print(f"\n> Clinical feature processing done (M = {config.NUM_CLINICAL_FEATURES})")
        
        X_train_feat.to_csv(f"{config.EXPORT_PATH}/{config.VERSION}/X_train_feat_fold{fold}.csv", index = False)
        X_test_feat.to_csv(f"{config.EXPORT_PATH}/{config.VERSION}/X_test_feat_fold{fold}.csv", index = False)
        
        sanity_check_feature_process(config, df_train, df_test, X_train_feat, X_test_feat)
        
        y_train, y_test = df_train[['GT']], df_test[['GT']]
            
        y_train.to_csv(f"{config.EXPORT_PATH}/{config.VERSION}/y_train_fold{fold}.csv", index = False)
        y_test.to_csv(f"{config.EXPORT_PATH}/{config.VERSION}/y_test_fold{fold}.csv", index = False)
        
        X_train_img = pd.DataFrame({
                'subject_number': df_train['subject_number'],
                'Visit Number': df_train['Visit Number'],
                'ICGUID': df_train['ICGUID'],
                'image': config.TRAIN_MSI + "/data/" + df_train['subject_number'] + "-" + df_train['ICGUID'] + ".npy",
                'mask': config.TRAIN_MSI + "/mask/" + df_train['subject_number'] + "-" + df_train['ICGUID'] + "_mask.npy",
                })
        train_dataset = DFUDataset(config,
                                   X_train_feat,
                                   X_train_img,
                                   y_train,
                                   indicator = 'train')
        
        X_test_img = pd.DataFrame({
                'subject_number': df_test['subject_number'],
                'Visit Number': df_test['Visit Number'],
                'ICGUID': df_test['ICGUID'],
                'image': config.TRAIN_MSI + "/data/" + df_test['subject_number'] + "-" + df_test['ICGUID'] + ".npy",
                'mask': config.TRAIN_MSI + "/mask/" + df_test['subject_number'] + "-" + df_test['ICGUID'] + "_mask.npy",
                })
        test_dataset = DFUDataset(config,
                                  X_test_feat,
                                  X_test_img,
                                  y_test,
                                  indicator = 'test')

        print(f"\n> Trained on {len(train_dataset)} ICGUIDs | tested on {len(test_dataset)} ICGUIDs\n")
        
        # Build the classification model
        if (config.MODEL.strip().lower() == 'baseline'):
            model = BaseLine(config)
        else:
            pass
            
        # Load pre-trained weights if available
        if config.PRETRAIN_PATH:
            model.load_state_dict(torch.load(config.PRETRAIN_PATH))
        
        set_seed(seed)
            
        train_and_validate(fold, config, model, train_dataset, test_dataset)
        
        np.save(f"{config.MODEL_SAVE_PATH}/config_final_train.npy", vars(config))
            
    else:
        
        for fold in range(1, config.K_FOLD + 1):
            
            print(f"\n------------------------------------------------------------------------------")
            print(f"Current fold: {fold}")
            
            cur_df_train = df_train[df_train['fold'] != fold].reset_index(drop = True)
            cur_df_val = df_train[df_train['fold'] == fold].reset_index(drop = True)
            
            preprocess = PreProcessing(config, cur_df_train, cur_df_val)
            X_train_feat, X_val_feat = preprocess.preprocess_df()
            
            config.NUM_CLINICAL_FEATURES = X_train_feat.shape[1]
            print(f"\n> Clinical feature processing done (M = {config.NUM_CLINICAL_FEATURES})")
            
            X_train_feat.to_csv(f"{config.EXPORT_PATH}/{config.VERSION}/X_train_feat_fold{fold}.csv", index = False)
            X_val_feat.to_csv(f"{config.EXPORT_PATH}/{config.VERSION}/X_val_feat_fold{fold}.csv", index = False)
            
            sanity_check_feature_process(config, cur_df_train, cur_df_val, X_train_feat, X_val_feat)

            y_train, y_val = cur_df_train[['GT']], cur_df_val[['GT']]
            
            y_train.to_csv(f"{config.EXPORT_PATH}/{config.VERSION}/y_train_fold{fold}.csv", index = False)
            y_val.to_csv(f"{config.EXPORT_PATH}/{config.VERSION}/y_val_fold{fold}.csv", index = False)
        
            X_train_img = pd.DataFrame({
                'subject_number': cur_df_train['subject_number'],
                'Visit Number': cur_df_train['Visit Number'],
                'ICGUID': cur_df_train['ICGUID'],
                'image': config.TRAIN_MSI + "/data/" + cur_df_train['subject_number'] + "-" + cur_df_train['ICGUID'] + ".npy",
                'mask': config.TRAIN_MSI + "/mask/" + cur_df_train['subject_number'] + "-" + cur_df_train['ICGUID'] + "_mask.npy",
                })
            train_dataset = DFUDataset(config,
                                       X_train_feat,
                                       X_train_img,
                                       y_train,
                                       indicator = 'train')
            
            X_val_img = pd.DataFrame({
                'subject_number': cur_df_val['subject_number'],
                'Visit Number': cur_df_val['Visit Number'],
                'ICGUID': cur_df_val['ICGUID'],
                'image': config.TRAIN_MSI + "/data/" + cur_df_val['subject_number'] + "-" + cur_df_val['ICGUID'] + ".npy",
                'mask': config.TRAIN_MSI + "/mask/" + cur_df_val['subject_number'] + "-" + cur_df_val['ICGUID'] + "_mask.npy",
                })
            val_dataset = DFUDataset(config,
                                     X_val_feat,
                                     X_val_img,
                                     y_val,
                                     indicator = 'test')
            
            print(f"\n> Trained on {len(train_dataset)} ICGUIDs | tested on {len(val_dataset)} ICGUIDs\n")
            
            # Build the classification model
            if (fold == 1):
                if (config.MODEL).strip().lower() == "baseline":
                    model = BaseLine(config)
                else:
                    pass
                initial_state = copy.deepcopy(model.state_dict())
            model.load_state_dict(initial_state)
            
            # Load pre-trained weights if available
            if config.PRETRAIN_PATH:
                model.load_state_dict(torch.load(config.PRETRAIN_PATH))
            
            set_seed(seed)
            
            train_and_validate(fold, config, model, train_dataset, val_dataset)
        
        np.save(f"{config.MODEL_SAVE_PATH}/config_CV.npy", vars(config))
        
    return None



def train_and_validate(fold, config, model, train_dataset, val_dataset):
    
    train_records, val_records = [], []
    
    # Build data loader
    train_loader = DataLoader(train_dataset,
                              batch_size = int(config.BATCH_SIZE),
                              shuffle = True,
                              num_workers = config.NUM_WORKERS,
                              drop_last = True)
    val_loader = DataLoader(val_dataset,
                            batch_size = int(config.BATCH_SIZE),
                            shuffle = False,
                            num_workers = config.NUM_WORKERS)
    
    model = model.to(config.DEVICE)
    
    # Define cost function
    if (config.CLF_LOSS['LOSS_NAME']).lower() == 'weighted_ce':
        cost_fn = weighted_CE
    else:
        raise Exception(f"ERROR: loss function not found in utils: {config.LOSS['LOSS_NAME']}")
    criterion = cost_fn(config).to(config.DEVICE)
    
    # Define optimizer
    if (config.OPTIMIZER_PARAMS["OPTIMIZER"] == 'adam'):
        optimizer = torch.optim.Adam(model.parameters(), 
                                     lr = config.OPTIMIZER_PARAMS["BASE_LR"])
    elif (config.OPTIMIZER_PARAMS["OPTIMIZER"] == 'sgd'):
        optimizer = torch.optim.SGD(model.parameters(),
                                    lr = config.OPTIMIZER_PARAMS["BASE_LR"],
                                    momentum = config.OPTIMIZER_PARAMS["MOMENTUM"],
                                    weight_decay = config.OPTIMIZER_PARAMS["WEIGHT_DECAY"])
    else:
        raise Exception(f"ERROR: optimizer not found in utils: {config.OPTIMIZER_PARAMS['OPTIMIZER']}")
    
    # Define learning rate scheduler
    scheduler = None
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
    #                                                        mode = "min", 
    #                                                        factor = 0.1, 
    #                                                        patience = 10, 
    #                                                        threshold = 0.01, 
    #                                                        threshold_mode = 'rel',
    #                                                        eps = 1e-6, 
    #                                                        verbose = False)
    
    save_path = f"{config.MODEL_SAVE_PATH}/snapshot_fold_{fold}"
    if not os.path.exists(save_path): os.makedirs(save_path)
        
    tic = time.time() # record time duration for all epochs
    for epoch in range(config.EPOCH):
        
        tic_ = time.time() # record time duration for individual epoch
        
        model, columns, train_performance = train(config, model, train_loader, criterion, epoch, optimizer)
        
        train_records.append(train_performance)
        
        val_performance = validate(config, val_loader, model, criterion, epoch, fold, scheduler)
        
        val_records.append(val_performance)
        
        if (config.SAVE_MODEL_PER_EPOCH): 
            torch.save(model.state_dict(), f"{save_path}/epoch_{epoch}.pth")

        print(f"Time spent: {round(time.time() - tic_)} sec || Current learning rate: {optimizer.state_dict()['param_groups'][0]['lr']}\n")
        
    # Export train/validation performance in each epoch
    export_path = f"{config.PREDICTION_PATH}/train_val_curve"
    if not os.path.exists(export_path): os.makedirs(export_path)
    
    df_train_records = pd.DataFrame(np.array(train_records), columns = columns)
    df_train_records.to_csv(os.path.join(export_path, "df_train_records" + str(fold) + ".csv"), index = False)
    
    df_val_records = pd.DataFrame(np.array(val_records), columns = columns)
    df_val_records.to_csv(os.path.join(export_path, "df_val_records" + str(fold) + ".csv"), index = False)
    
    toc = time.time()
    
    print(f"Training completed (Execution time: {(toc - tic) / 60:.2f} minutes)")
    
    return None
    
        

def train(config, model, train_loader, criterion, epoch, optimizer):
    
    mm = MetricMonitor()
    
    tot_output, tot_target, tot_sz = torch.tensor([]).to(config.DEVICE), torch.tensor([]).to(config.DEVICE), 0
    
    model.train() # train mode starts
    
    for _, (clin_features, images, mask, target, _) in enumerate(train_loader, start = 1):

        # Get current input
        clin_features = clin_features.to(torch.float32).to(config.DEVICE)
        images = images.to(config.DEVICE) # shape: (batch_size, C, H, W)
        mask = mask.to(config.DEVICE) # shape: (batch_size, 1, H, W)
        target = torch.squeeze(target).long().to(config.DEVICE)
        
        # Forward propagation and compute loss
        output = model(clin_features, images, mask)
        loss = criterion(output, target)
        
        # Back propagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        tot_output = torch.cat((tot_output, output), dim = 0)
        tot_target = torch.cat((tot_target, target), dim = 0)
        tot_sz += float(images.shape[0])
    
    # Get loss for current epoch
    loss_tot = criterion(tot_output, tot_target.long())
    mm.update("loss", loss_tot.item() * tot_sz, tot_sz)
    
    # Get performance for current epoch
    tot_pred = F.softmax(tot_output, dim = 1) # convert to probabilities
    tot_pred = tot_pred[:, 1]
    for metric_fn in [clf_metrics.acc, clf_metrics.sen, clf_metrics.spe, clf_metrics.harmonic]:
        val, ct = metric_fn(tot_pred, tot_target, thres = 0.5)
        mm.update(metric_fn.__name__, val, ct)
        
    print(f"Epoch {epoch} / {config.EPOCH - 1} Training performance: | loss: {mm.value('loss'):.3f} | acc: {mm.value('acc'):.3f} | sen: {mm.value('sen'):.3f} | spe: {mm.value('spe'):.3f} | harmonic mean: {mm.value('harmonic'):.3f}")
    
    columns, performance = ['loss'], [mm.value('loss')]
    for metric_fn in [clf_metrics.acc, clf_metrics.sen, clf_metrics.spe, clf_metrics.harmonic]:
        columns.append(metric_fn.__name__)
        performance.append(mm.value(metric_fn.__name__))

    return model, columns, performance


    
def validate(config, val_loader, model, criterion, epoch, fold, scheduler = None):
    
    mm = MetricMonitor()
    
    tot_pred, tot_target = torch.tensor([]).to(config.DEVICE), torch.tensor([]).to(config.DEVICE)
    
    model.eval() # eval mode starts
    
    save_path = f"{config.PREDICTION_PATH}/fold_{fold}/"
    if not os.path.exists(save_path): os.makedirs(save_path)
    
    res_df = pd.DataFrame(columns = ['subject_number', 'Visit Number', 'ICGUID', 'Pred_Proba', 'GT'])
    
    with torch.no_grad():
        
        for _, (clin_features, images, mask, target, (subject_number, visit_number, icguid)) in enumerate(val_loader, start = 1):
            
            cur_batch_sz = float(images.shape[0])
            
            # Get current input
            clin_features = clin_features.to(torch.float32).to(config.DEVICE)
            images = images.to(config.DEVICE) # shape: (batch_size, C, H, W)
            mask = mask.to(config.DEVICE)
            target = torch.squeeze(target).long().to(config.DEVICE)
            if target.ndim == 0: target = target.unsqueeze(0)
            
            # Forward propagation and compute loss
            output = model(clin_features, images, mask)
            loss = criterion(output, target)
            mm.update('loss', loss.item() * cur_batch_sz, cur_batch_sz)
            
            # Get model predictions with other suppl. info
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
            
            tot_pred = torch.cat((tot_pred, pred), dim = 0) 
            tot_target = torch.cat((tot_target, target), dim = 0)
    
    res_df.to_csv(f"{save_path}/pred_epoch_{epoch}.csv", index = False)
    
    for metric_fn in [clf_metrics.acc, clf_metrics.sen, clf_metrics.spe, clf_metrics.harmonic]:
        val, ct = metric_fn(tot_pred, tot_target, thres = 0.5)
        mm.update(metric_fn.__name__, val, ct)
                
    if scheduler: scheduler.step(mm.value('loss'))
    
    print(f"Validation performance: | loss: {mm.value('loss'):.3f} | acc: {mm.value('acc'):.3f} | sen: {mm.value('sen'):.3f} | spe: {mm.value('spe'):.3f} | harmonic mean: {mm.value('harmonic'):.3f}")
    
    performance = [mm.value('loss')]
    for metric_fn in [clf_metrics.acc, clf_metrics.sen, clf_metrics.spe, clf_metrics.harmonic]:
        performance.append(mm.value(metric_fn.__name__))
    
    return performance
    
                      
            
def print_exp_info(config):
    
    print(f"\n> Input Data Processing")
    print("    >> Clinical feature processing:")
    for key, val in config.NUMERICAL_PREPROCESS.items():
        print(f"    {key.lower()}: {val}")
    print("    >> Image processing:")
    for key, val in config.IMAGE_PREPROCESS.items():
        print(f"        {key.lower()}: {val}")
    
    print(f"\n> Model Info")
    print(f"    >> Algorithm: {config.MODEL}")
    print(f"    >> Hyperparameter set: {config.VERSION}")
    print("    >> Model parameters:")
    for key, val in config.MODEL_PARAMETERS.items():
        print(f"        {key.lower()}: {val}")
    print(f"    >> Batch size: {config.BATCH_SIZE}")
    print(f"    >> Epochs: {config.EPOCH}")
    print(f"    >> Export path: {config.EXPORT_PATH}")
    
    print(f"\n Loss/Optimizer Info")
    print("    >> Loss function:")
    for key, val in config.CLF_LOSS.items():
        print(f"        {key.lower()}: {val}")
    print("    >> Optimizer:")
    for key, val in config.OPTIMIZER_PARAMS.items():
        print(f"        {key.lower()}: {val}")
    
    return None



def set_seed(seed):
    random.seed(seed)   
    np.random.seed(seed)   
    torch.manual_seed(seed)           # Sets the seed for generating random numbers.
    torch.cuda.manual_seed(seed)      # Sets the seed for generating random numbers for the current GPU
    torch.cuda.manual_seed_all(seed)  # Sets the seed for generating random numbers on all GPUs.
    torch.backends.cudnn.deterministic = True # Allow cuDNN to deterministically select an algorithm to increase performance



def sanity_check_feature_process(config, df_train, df_val, df_train_transformed_in, df_val_transformed_in):
    """
    Helper function: Sanity check the correctness of clinical feature preprocessing
    
    Input:
    config - (Class) configuration for model training
    df_train - (DataFrmae) features from the training set before processing
    df_val - (DataFrame) features from the validation/test set before processing
    df_train_transformed_in - (DataFrame) processed features from the training set
    df_val_transformed_in - (DataFrame) processed features from the validation/test set
    """
    
    print(f"\n> Sanity check processed clinical features")
    
    df_train_cp, df_val_cp = df_train.copy(), df_val.copy()
    df_train_transformed = df_train_transformed_in.copy()
    df_val_transformed = df_val_transformed_in.copy()
    
    preprocess = PreProcessing(config, df_train_cp, df_val_cp)
    
    # Get list of numerical features
    numerical_list = preprocess.NUMERICAL_1 + preprocess.NUMERICAL_2 + preprocess.NUMERICAL_3
    numerical_list.extend(['subject_number', 'Visit Number'])
    X_train_numerical = df_train_cp.loc[:, numerical_list].copy()
    
    numerical_stats = {}
    for col in numerical_list:
        
        if (col == 'subject_number') or (col == 'Visit Number'): continue
        
        mean_ = X_train_numerical.drop_duplicates()[col].mean()
        std_ = X_train_numerical.drop_duplicates()[col].std()
        numerical_stats[f"train_mean_{col}"] = mean_
        numerical_stats[f"train_std_{col}"] = std_
        
        min_ = X_train_numerical.drop_duplicates()[col].min()
        max_ = X_train_numerical.drop_duplicates()[col].max()
        numerical_stats[f"train_min_{col}"] = min_
        numerical_stats[f"train_max_{col}"] = max_
        
    # Get list of categorical features
    categorical_list = preprocess.CAT_BIN_2 + list(preprocess.CAT_ONEHOT.keys())
    categorical_list.extend(['subject_number', 'Visit Number'])
    X_train_categorical = df_train_cp.loc[:, categorical_list].copy()
    
    categorical_stats = {}
    for col in categorical_list:
        
        if (col == 'subject_number') or (col == 'Visit Number'): continue
        
        mode_ = X_train_categorical.drop_duplicates()[col].mode().iloc[0]
        categorical_stats[f"train_mode_{col}"] = mode_

    for df in [df_train_cp, df_val_cp]:
        
        is_train = True if df.equals(df_train_cp) else False
        
        for col in df.columns:
            
            for idx in range(len(df)):
                
                subjectID = df.loc[idx, 'subject_number']
                visit = df.loc[idx, 'Visit Number'][4:]
                
                res = df.loc[idx, col] # fetch the current cell value
                
                ##############################################################
                # Check 1 - Numerical features
                ##############################################################
                if (col in preprocess.NUMERICAL_1) or (col in preprocess.NUMERICAL_2) or (col in preprocess.NUMERICAL_3):
                    
                    ref = df_train_transformed.loc[idx, col] if is_train else df_val_transformed.loc[idx, col]
                    
                    if (np.isnan(res)):
                        ref = ref * numerical_stats[f'train_std_{col}'] + numerical_stats[f'train_mean_{col}']
                        print_str = f"Imputed feature {col} for subjectID {subjectID} - {visit}: {ref:.3f} (min: {numerical_stats[f'train_min_{col}']:.3f}, max: {numerical_stats[f'train_max_{col}']:.3f})"
                        if (is_train):
                            print(f"    >> TRAIN SET - {print_str}")
                        else:
                            print(f"    >> VAL SET - {print_str}")
                    else:
                        res = (res - numerical_stats[f"train_mean_{col}"]) / numerical_stats[f"train_std_{col}"] # Normalization
                        if (abs(res - ref) > 0.1): 
                            print(f"    >> WARNING: Potential mismatch found in feature {col} for subjectID {subjectID} - {visit}: result: {ref:.3f} | expected: {res:.3f}")
                    
                ##############################################################
                # Check 2 - Categorical features
                ##############################################################    
                if (col in preprocess.CAT_BIN_1):
                    
                    ref = df_train_transformed.loc[idx, col] if is_train else df_val_transformed.loc[idx, col]
                    
                    if str(res) == 'nan':
                        res = 0
                        print_str = f"Imputed feature {col} for subjectID {subjectID} - {visit}: {ref}"
                        if (is_train):
                            print(f"    >> TRAIN SET - {print_str}")
                        else:
                            print(f"    >> VAL SET - {print_str}")
                    else:
                        if res.strip().lower() == 'yes':
                            res = 1
                        elif res.strip().lower() == 'no':
                            res = 0
                        else:
                            raise Exception(f"    >> ERROR: unknown value found in feature {col} for subjectID {subjectID} - {visit}")
    
                    if (res != ref): raise Exception(f"    >> ERROR: Mismatch found in feature {col} for subjectID {subjectID} - {visit}: result: {ref} | expected: {res}")
                    
                if (col in preprocess.CAT_BIN_2):
                    
                    ref = df_train_transformed.loc[idx, col] if is_train else df_val_transformed.loc[idx, col]
                    
                    if (str(res) == 'nan'):
                        res = categorical_stats[f"train_mode_{col}"]
                        print_str = f"Imputed feature {col} with mode value for subjectID {subjectID} - {visit}: {ref}"
                        if (is_train):
                            print(f"    >> TRAIN SET - {print_str}")
                        else:
                            print(f"    >> VAL SET - {print_str}")
                    else:
                        if (res.strip().lower() == 'yes') or (res.strip().lower() == 'male'):
                            res = 1
                        elif (res.strip().lower() == 'no') or (res.strip().lower() == 'female'):
                            res = 0
                        else:
                            raise Exception(f"    >> ERROR: unknown value found: {res} when processing feature {col}")
                        
                    if (res != ref): raise Exception(f"    >> ERROR: Mismatch found in feature {col} for subjectID {subjectID} - {visit}: result: {ref} | expected: {res}")
                    
                if (col in preprocess.CAT_ONEHOT):
                    
                    unique_entries = preprocess.CAT_ONEHOT[col]
                    
                    if (str(res) == 'nan'):
                        res = categorical_stats[f"train_mode_{col}"]
                        print_str = f"Imputed feature {col} with mode value for subjectID {subjectID} - {visit}: {res}"
                        if (is_train):
                            print(f"    >> TRAIN SET - {print_str}")
                        else:
                            print(f"    >> VAL SET - {print_str}")
                    
                    for entry in unique_entries:
                         
                        ref = df_train_transformed.loc[idx, f"{col}_{entry}"] if is_train else df_val_transformed.loc[idx, f"{col}_{entry}"]

                        if (entry == res and ref != 1) or (entry != res and ref != 0):            
                            raise Exception(f"    >> ERROR: Mismatch found in feature {col} for subjectID {subjectID} | expected: {res}")

                if (col in preprocess.ORDINAL):
                    
                    ref = df_train_transformed.loc[idx, col] if is_train else df_val_transformed.loc[idx, col]
                    
                    if str(res) == 'nan':
                        res = 0
                        print_str = f"Imputed feature {col} for subjectID {subjectID} - {visit}: {ref}"
                        if (is_train):
                            print(f"    >> TRAIN SET - {print_str}")
                        else:
                            print(f"    >> VAL SET - {print_str}")
                        if (res != ref):
                            raise Exception(f"    >> ERROR: Mismatch found in feature {col} for subjectID {subjectID}: result: {ref} | expected: {res:.3f}")
                    else:
                        if (preprocess.ORDINAL[col][res] != ref):
                            raise Exception(f"    >> ERROR: Mismatch found in feature {col} for subjectID {subjectID}: result: {ref} | expected: {res:.3f}")