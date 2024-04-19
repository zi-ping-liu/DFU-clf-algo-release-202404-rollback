"""
Customized Dataset class for processing clinical features and multi-spectral images.

Author: Ziping Liu
Date: Apr 19, 2024
"""



# Import libraries
from torch.utils.data import Dataset
import numpy as np
import albumentations as albu
import cv2
import torch



class DFUDataset(Dataset):
    
    
    def __init__(self, config, X_feat, X_img, y, indicator):
        
        self.config = config
        self.X_feat = X_feat
        self.X_img = X_img
        self.y = y
        self.indicator = indicator
        
        
    def __len__(self):
        
        return len(self.X_feat)
    
    
    def __getitem__(self, idx):
        
        subject_number = self.X_img.iloc[idx]['subject_number']
        
        icguid = self.X_img.iloc[idx]['ICGUID']
        
        visit_number = self.X_img.iloc[idx]['Visit Number']
        
        clin_features = torch.tensor(self.X_feat.iloc[idx])
        
        target = torch.tensor(self.y.iloc[idx])
        
        msi_img = np.load(self.X_img.iloc[idx]['image'], allow_pickle = True)
        
        mask = np.load(self.X_img.iloc[idx]['mask'], allow_pickle = True)
        mask = mask[:, :, np.newaxis] # (H, W, C)
        
        if (self.config.MODEL_PARAMETERS['attention_type'] == 0):
            msi_img = msi_img * mask
        
        if (self.config.IMAGE_PREPROCESS['AUGMENTATION'] == 'v1'):
            augm = apply_augmentation_v1(self.config, self.indicator)
        else:
            augm = apply_augmentation_baseline(self.config)
            
        sample = augm(image = msi_img, mask = mask)
        msi_img, mask = sample['image'], sample['mask']
        
        preproc = apply_preprocessing(self.config)
        sample = preproc(image = msi_img, mask = mask)
        msi_img, mask = sample['image'], sample['mask']
        
        return clin_features, msi_img, mask, target, (subject_number, visit_number, icguid)   
    
    
    
def apply_augmentation_v1(config, indicator = 'train'):
    """
    Perform image augmentation by applying the following operations randomly:
    (a) Grid distortion: modeling images that could be taken from different angles or distances, or under different lighting conditions
    (b) Optical distortion: modeling image distortion that occurs in camera lenses, such as barrel or pincushion distortion
    (c) Shift, scale, rotation
    (d) Horizontal flip
    (e) Vertical flip
    """
    
    # Resizing images (if applicable)
    transform = [
        albu.Resize(height = config.IMAGE_PREPROCESS['IMAGE_SIZE'][0], width = config.IMAGE_PREPROCESS['IMAGE_SIZE'][1])
        ]
        
    if indicator == "train":
        transform.extend([
            albu.GridDistortion(border_mode = 0, value = 0, interpolation = cv2.INTER_LANCZOS4, p = 0.5),
            albu.OpticalDistortion(distort_limit = 0.05, shift_limit = 0.05, p = 0.5),
            albu.ShiftScaleRotate(shift_limit = 0.0625, scale_limit = 0.1, rotate_limit = 180, border_mode = 0, value = 0, p = 0.5),
            albu.HorizontalFlip(p = 0.5),
            albu.VerticalFlip(p = 0.5)
        ])
        
    return albu.Compose(transform)



def apply_augmentation_baseline(config):
    """
    No augmentation performed. Only resizing images (if applicable)
    """
    
    transform = [
        albu.Resize(height = config.IMAGE_PREPROCESS["IMAGE_SIZE"][0], width = config.IMAGE_PREPROCESS["IMAGE_SIZE"][1])
        ]
        
    return albu.Compose(transform)



def apply_preprocessing(config):

    transform = []
        
    if config.IMAGE_PREPROCESS['NORMALIZATION'] == 'standard_local':
        transform.extend([
            albu.Lambda(image = standardize_local),
            albu.Lambda(image = transpose, mask = transpose),
        ])
        
    elif config.IMAGE_PREPROCESS['NORMALIZATION'] == 'standard_global':
        transform.extend([
            albu.Lambda(image = standardize_global),
            albu.Lambda(image = transpose, mask = transpose),
        ])
        
    elif config.IMAGE_PREPROCESS['NORMALIZATION'] == 'min_max_local':
        transform.extend([
            albu.Lambda(image = normalize_local),
            albu.Lambda(image = transpose, mask = transpose),
        ])
        
    elif config.IMAGE_PREPROCESS['NORMALIZATION'] == 'min_max_global':
        transform.extend([
            albu.Lambda(image = normalize_global),
            albu.Lambda(image = transpose, mask = transpose),
        ])
        
    else:
        transform.extend([
            albu.Lambda(image = transpose, mask = transpose)
        ])
        
    return albu.Compose(transform)



def standardize_local(image, **kwargs):
    # image standardization within each channel
    mean_, std_ = np.mean(image, axis = (0, 1)), np.std(image, axis = (0, 1))
    return (image - mean_ + 1e-7) / (std_ + 1e-7)



def standardize_global(image, **kwargs):
    # image standardization across all channels
    mean_, std_ = np.mean(image), np.std(image)
    return (image - mean_ + 1e-7) / (std_ + 1e-7)



def normalize_local(image, **kwargs):
    # min-max normalization within each channel
    min_, max_ = np.min(image, axis = (0, 1)), np.max(image, axis = (0, 1))
    return (image - min_) / (max_ - min_)



def normalize_global(image, **kwargs):
    # min-max normalization across all channels
    min_, max_ = np.min(image), np.max(image)
    return (image - min_) / (max_ - min_)



def transpose(x, **kwargs):
    return x.transpose(2, 0, 1).astype("float32") # (H, W, C) -> (C, H, W)