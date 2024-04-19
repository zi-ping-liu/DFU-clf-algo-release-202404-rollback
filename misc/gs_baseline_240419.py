"""
Generate grid-search CSV for K-fold cross-validation of baseline model

Author: Ziping Liu
Date: Apr 19, 2024
"""



# Import libraries
import os
import pandas as pd
from datetime import date

export_path = "/home/efs/ziping/workspaces/dfu/clf_algo_release_202404_rollback/src/data/grid_search_baseline_240419.csv"
if os.path.exists(export_path):
    raise Exception(f"A grid search look-up table with the same name already exists: {export_path}")

df = pd.DataFrame()

ct = 0

for batch_size in [4, 8]:
    
    for dropout_prob in [0.25, 0.5]:
        
        for lr in [1e-4, 1e-5, 1e-6]:
            
            for encoder_filters in [[16, 32, 64], [16, 32, 64, 128]]:
                
                for num_group_norm in [4, 8, 16]:
                    
                    for attention_type in [1]:
                        
                        for image_preprocess_normalization in ['min_max_local', 'min_max_global', 'standard_global']:
                            
                            for num_conv in [1]:
                            
                                for class_weight in [[1., 2.], [1., 4.]]:
                                    
                                    for imag_fcn in [[64], [128], [256]]:
                                                
                                        if imag_fcn == [64]:
                                            mixed_fcn = [32]
                                        elif imag_fcn == [128]:
                                            mixed_fcn = [64]
                                        else:
                                            mixed_fcn = [128]
                                            
                                        ct += 1
                                        
                                        entry = {
                                                'version': f"hs_{ct}",
                                                
                                                'use_good_ori_only': True,
                                                
                                                'numerical_preprocess_normalization': 'standard',
                                                'numerical_preprocess_num_nearest_neighbor': 5,
                                                
                                                'image_preprocess_image_size': [700, 700],
                                                'image_preprocess_normalization': image_preprocess_normalization,
                                                'image_preprocess_augmentation': '',
                                                'image_preprocess_mask_dilation': 0,
                                                
                                                'dfu_classes': 2,
                                                'attention_type': attention_type,
                                                'encoder_filters': encoder_filters,
                                                'encoder_kernel_size': 3,
                                                'encoder_dropout_prob': dropout_prob,
                                                'encoder_normalization': 'group',
                                                'num_group_norm': num_group_norm,
                                                'num_conv': num_conv,
                                                'ds_stride': 2,
                                                'imag_fcn': imag_fcn,
                                                'imag_dropout_prob': dropout_prob,
                                                'clin_fcn': [64, 32],
                                                'clin_dropout_prob': dropout_prob,
                                                'mixed_fcn': mixed_fcn,
                                                'mixed_dropout_prob': dropout_prob,
                                                
                                                'loss_name': 'weighted_CE',
                                                'class_weight': class_weight,
                                                
                                                'lr': lr,
                                                
                                                'batch_size': batch_size,
                                                'epoch_size': 30
                                        }
                                        
                                        df = pd.concat([df, pd.DataFrame([entry])], ignore_index = True)
                                    
df.to_csv(export_path, index = False)
