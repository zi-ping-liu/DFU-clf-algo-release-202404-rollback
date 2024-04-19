"""
Metrics used to measure classification performance

Author: Ziping Liu
Date: Apr 19, 2024
"""



# Import libraries
import torch



def acc(pred, target, thres = 0.5):
    
    yp = pred.clone().to(torch.float32)
    y = target.clone().to(torch.float32)
    
    yp = (yp > thres).type(yp.dtype)

    TP = torch.sum((yp == 1) & (y == 1)).item()
    FP = torch.sum((yp == 1) & (y == 0)).item()
    TN = torch.sum((yp == 0) & (y == 0)).item()
    FN = torch.sum((yp == 0) & (y == 1)).item()
            
    acc_ = (TP + TN) / (TP + FP + TN + FN)
            
    return acc_, 1


def sen(pred, target, thres = 0.5):
    
    yp = pred.clone().to(torch.float32)
    y = target.clone().to(torch.float32)
    
    yp = (yp > thres).type(yp.dtype)

    TP = torch.sum((yp == 1) & (y == 1)).item()
    FP = torch.sum((yp == 1) & (y == 0)).item()
    TN = torch.sum((yp == 0) & (y == 0)).item()
    FN = torch.sum((yp == 0) & (y == 1)).item()
            
    sen_ = (TP) / (TP + FN)
            
    return sen_, 1


def spe(pred, target, thres = 0.5):
    
    yp = pred.clone().to(torch.float32)
    y = target.clone().to(torch.float32)
    
    yp = (yp > thres).type(yp.dtype)

    TP = torch.sum((yp == 1) & (y == 1)).item()
    FP = torch.sum((yp == 1) & (y == 0)).item()
    TN = torch.sum((yp == 0) & (y == 0)).item()
    FN = torch.sum((yp == 0) & (y == 1)).item()
            
    spe_ = (TN) / (FP + TN)
            
    return spe_, 1


def harmonic(pred, target, thres = 0.5, eps = 1e-8):
    
    yp = pred.clone().to(torch.float32)
    y = target.clone().to(torch.float32)
    
    yp = (yp > thres).type(yp.dtype)

    TP = torch.sum((yp == 1) & (y == 1)).item()
    FP = torch.sum((yp == 1) & (y == 0)).item()
    TN = torch.sum((yp == 0) & (y == 0)).item()
    FN = torch.sum((yp == 0) & (y == 1)).item()
    
    acc_ = (TP + TN + eps) / (TP + FP + TN + FN)
    
    sen_ = (TP + eps) / (TP + FN)
    
    spe_ = (TN + eps) / (FP + TN)
    
    hm = 3 / (1 / acc_ + 1 / sen_ + 1 / spe_)
            
    return hm, 1
