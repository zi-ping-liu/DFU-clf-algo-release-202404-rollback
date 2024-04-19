"""
Loss functions used for classification
Note: ensure that config is the only input to your customized function 

Author: Ziping Liu
Date: Apr 19, 2024
"""



# Import libraries
import torch



def weighted_CE(config):
    """
    Weighted cross-entropy loss
    """
    
    class_weights = torch.tensor(config.CLF_LOSS["CLASS_WEIGHTS"], dtype = torch.float32)
    
    criterion = torch.nn.CrossEntropyLoss(weight = class_weights)
    
    return criterion