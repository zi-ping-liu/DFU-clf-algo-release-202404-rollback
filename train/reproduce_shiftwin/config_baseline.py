"""
Configuration for baseline model training with K-fold cross-validation

Author: Ziping Liu
Date: Apr 19, 2024
"""



# Load libraries
import warnings
warnings.filterwarnings("ignore")
import sys
sys.path.append("/home/efs/ziping/workspaces/dfu/clf_algo_release_202404_rollback/src")
from main_v0 import main
import pandas as pd
import json



class Config():
    
    
    def __init__(self,
                 version, # used to track hyperparameter sets (e.g., hs_1)
                 gpu_device, # assigned GPU device
                 train_csv, # path to unified CSV for WAUSI training study
                 train_msi, # path to WAUSI MSI images/masks
                 train_set, # list containing train subjectIDs
                 test_set, # list containing test subjectIDs
                 k_fold, # number of training folds; if k_fold == 1, final train is performed
                 numerical_preprocess, # dictionary containing all relevant parameters for preprocessing numerical features
                 image_preprocess, # dictionary containing all relevant parameters for preprocessing MSI images
                 model_params, # dictionary containing all relevant model parameters
                 pretrain_path, # path to pretrained model weights if applicable
                 clf_loss, # dictionary containing all relevant loss parameters
                 batch_sz, # batch size (e.g., 4)
                 epoch_sz, # epoch size (e.g., 50)
                 optimizer_params, # dictionary containing all relevant optimizer parameters
                 save_model_per_epoch # if true, .pth file is generated for each epoch 
                 ):
        
        #########################################################################
        #                              GPU SET UP
        self.DEVICE = gpu_device
        self.NUM_WORKERS = 12
        
        
        #########################################################################
        #                           DATA PREPARATION
        self.TRAIN_CSV = train_csv
        
        self.TRAIN_MSI = train_msi
        
        self.TRAIN_SET = train_set
        
        self.TEST_SET = test_set
        
        self.K_FOLD = k_fold
        
        self.NUMERICAL_PREPROCESS = numerical_preprocess
        
        self.IMAGE_PREPROCESS = image_preprocess
        
        
        #########################################################################
        #                         MODEL HYPERPARAMETER
        self.VERSION = version
        
        self.MODEL = 'baseline'
        
        self.MODEL_PARAMETERS = model_params
        
        self.PRETRAIN_PATH = pretrain_path
        
        self.CLF_LOSS = clf_loss
        
        
        #########################################################################
        #                            TRAINING DETAIL
        self.EXPORT_PATH = "/home/efs/ziping/workspaces/dfu/clf_algo_release_202404_rollback/results/reproduce_shiftwin/" + self.MODEL.strip().lower()

        self.BATCH_SIZE = batch_sz
        
        self.EPOCH = epoch_sz
        
        self.OPTIMIZER_PARAMS = optimizer_params
        
        self.SAVE_MODEL_PER_EPOCH = save_model_per_epoch
        
        
        
if __name__ == "__main__":
    
    gpu_device = int(sys.argv[1])
    
    idx = int(sys.argv[2])
    
    final_train = sys.argv[3]
    if final_train.strip().lower() == 'true':
        k_fold = 1
        save_model_per_epoch = True
    else:
        k_fold = 5
        save_model_per_epoch = False
    
    gs_df = pd.read_csv("/home/efs/ziping/workspaces/dfu/clf_algo_release_202404_rollback/src/data/grid_search_021024.csv")
    numerical_preprocess = {
        'NORMALIZATION': gs_df.iloc[idx]['numerical_preprocess_normalization'],
        'NUM_NEAREST_NEIGHBOURS': gs_df.iloc[idx]['numerical_preprocess_num_nearest_neighbor']
    }
    
    image_preprocess = {
        'USE_GOOD_ORI_ONLY': gs_df.iloc[idx]['use_good_ori_only'],
        'IMAGE_SIZE': json.loads(gs_df.iloc[idx]['image_preprocess_image_size']),
        'NORMALIZATION': gs_df.iloc[idx]['image_preprocess_normalization'],
        'AUGMENTATION': gs_df.iloc[idx]['image_preprocess_augmentation'],
        'MASK_DILATION': gs_df.iloc[idx]['image_preprocess_mask_dilation']
    }
    
    model_params = {
        'dfu_classes': gs_df.iloc[idx]['dfu_classes'],
        'attention_type': gs_df.iloc[idx]['attention_type'],
        'msi_dim': image_preprocess['IMAGE_SIZE'][0],
        'msi_channels': 8,
        'encoder_filters': json.loads(gs_df.iloc[idx]['encoder_filters']),
        'encoder_kernel_size': gs_df.iloc[idx]['encoder_kernel_size'],
        'encoder_normalization': gs_df.iloc[idx]['encoder_normalization'],
        'encoder_dropout_prob': gs_df.iloc[idx]['encoder_dropout_prob'],
        'num_group_norm': gs_df.iloc[idx]['num_group_norm'],
        'num_conv': gs_df.iloc[idx]['num_conv'],
        'ds_stride': gs_df.iloc[idx]['ds_stride'],
        'imag_fcn': json.loads(gs_df.iloc[idx]['imag_fcn']),
        'imag_dropout_prob': gs_df.iloc[idx]['imag_dropout_prob'],
        'clin_fcn': json.loads(gs_df.iloc[idx]['clin_fcn']),
        'clin_dropout_prob': gs_df.iloc[idx]['clin_dropout_prob'],
        'mixed_fcn': json.loads(gs_df.iloc[idx]['mixed_fcn']),
        'mixed_dropout_prob': gs_df.iloc[idx]['mixed_dropout_prob']
    }
    
    clf_loss = {
        'LOSS_NAME': gs_df.iloc[idx]['loss_name'],
        'CLASS_WEIGHTS': json.loads(gs_df.iloc[idx]['class_weight'])
    }
    
    optimizer_params = {
        'OPTIMIZER': 'adam',
        'BASE_LR': gs_df.iloc[idx]['lr'],
    }
    
    train_csv = "/home/efs/ziping/workspaces/dfu/clf_algo_release_202404_rollback/src/data/processed_WAUSI_BSV+shiftwin_20240304.csv"
    df = pd.read_csv(train_csv)
    
    if image_preprocess['USE_GOOD_ORI_ONLY']: # Keep only good-orientation cases
        df = df[df['good_ori'] == 'Y'].reset_index(drop = True)
    
    train_set = df[df['DS_split'] == 'train']['subject_number'].unique().tolist()
    test_set = df[df['DS_split'] != 'train']['subject_number'].unique().tolist()
    
    train_msi = f"/home/efs/TrainingData/DFU/BSV+slidingwindow_Processed_V4_centercrop_20240304/cropped_msi_ulcer_only/morph_{image_preprocess['MASK_DILATION']}"
    
    config = Config(
            version = gs_df.iloc[idx]['version'],
            gpu_device = gpu_device,
            train_csv = train_csv,
            train_msi = train_msi,
            train_set = train_set,
            test_set = test_set,
            k_fold = k_fold,
            numerical_preprocess = numerical_preprocess,
            image_preprocess = image_preprocess,
            pretrain_path = '',
            clf_loss = clf_loss,
            model_params = model_params,
            batch_sz = gs_df.iloc[idx]['batch_size'],
            epoch_sz = gs_df.iloc[idx]['epoch_size'],
            optimizer_params = optimizer_params,
            save_model_per_epoch = save_model_per_epoch
    )
    
    main(config)