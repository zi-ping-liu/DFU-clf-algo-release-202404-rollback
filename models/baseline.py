#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
DFU 28-days wound healing prediction - Baseline model

Author: Ziping Liu
Date: 03/01/24
"""



# Import libraries
import torch
import torch.nn as nn
import torch.nn.functional as F



class BaseLine(nn.Module):
    
    
    def __init__(self, config):
        
        super().__init__()
        
        self.config = config
        
        # Layer 1
        self.conv1, self.outW_conv1  = self.conv_layer(in_W = self.config.MODEL_PARAMETERS['msi_dim'],
                                                       in_C = self.config.MODEL_PARAMETERS['msi_channels'],
                                                       out_C =  self.config.MODEL_PARAMETERS['encoder_filters'][0])
        self.ds1, self.outW_ds1 = self.ds_layer(in_W = self.outW_conv1)
        
        # Layer 2
        self.conv2, self.outW_conv2  = self.conv_layer(in_W = self.outW_ds1,
                                                       in_C = self.config.MODEL_PARAMETERS['encoder_filters'][0],
                                                       out_C =  self.config.MODEL_PARAMETERS['encoder_filters'][1])
        self.ds2, self.outW_ds2 = self.ds_layer(in_W = self.outW_conv2)
        
        # Layer 3
        self.conv3, self.outW_conv3  = self.conv_layer(in_W = self.outW_ds2,
                                                       in_C = self.config.MODEL_PARAMETERS['encoder_filters'][1],
                                                       out_C = self.config.MODEL_PARAMETERS['encoder_filters'][2])
        self.ds3, self.outW_ds3 = self.ds_layer(in_W = self.outW_conv3)
        self.out_encoder_feats = (self.outW_ds3 ** 2) * self.config.MODEL_PARAMETERS['encoder_filters'][2]
        
        # Layer 4 (Optional)
        if (len(self.config.MODEL_PARAMETERS['encoder_filters']) > 3):
            self.conv4, self.outW_conv4  = self.conv_layer(in_W = self.outW_ds3,
                                                           in_C = self.config.MODEL_PARAMETERS['encoder_filters'][2],
                                                           out_C =  self.config.MODEL_PARAMETERS['encoder_filters'][3])
            self.ds4, self.outW_ds4 = self.ds_layer(in_W = self.outW_conv4)
            self.out_encoder_feats = (self.outW_ds4 ** 2) * self.config.MODEL_PARAMETERS['encoder_filters'][3] 
        
        # Mask-guided attention
        if (config.MODEL_PARAMETERS['attention_type'] == 1):
            padding = (config.MODEL_PARAMETERS['encoder_kernel_size'] - 1) // 2
            if (config.MODEL_PARAMETERS['encoder_kernel_size'] == 3):
                custom_filter = torch.tensor([[0, 0, 0],
                                              [0, 1, 0],
                                              [0, 0, 0]], dtype = torch.float32, requires_grad = False).unsqueeze(0).unsqueeze(0)
            elif (config.MODEL_PARAMETERS['encoder_kernel_size'] == 5):
                custom_filter = torch.tensor([[0, 0, 0, 0, 0],
                                              [0, 0, 0, 0, 0],
                                              [0, 0, 1, 0, 0],
                                              [0, 0, 0, 0, 0],
                                              [0, 0, 0, 0, 0]], dtype = torch.float32, requires_grad = False).unsqueeze(0).unsqueeze(0)
            else:
                raise Exception("ERROR: unknown kernel size || Please select from [3, 5]")
            
            custom_filter = custom_filter.cuda()
            self.attentionlayers = []
            for idx in range(len(config.MODEL_PARAMETERS['encoder_filters'])):         
                if (idx == 0):
                    in_C = 1
                    custom_filters = custom_filter.repeat(config.MODEL_PARAMETERS['encoder_filters'][idx], 1, 1, 1)
                else:
                    in_C = config.MODEL_PARAMETERS['encoder_filters'][idx - 1]
                    custom_filters = custom_filter.repeat(config.MODEL_PARAMETERS['encoder_filters'][idx], config.MODEL_PARAMETERS['encoder_filters'][idx - 1], 1, 1)
                out_C = config.MODEL_PARAMETERS['encoder_filters'][idx]
                
                self.attentionlayer = nn.Conv2d(in_C, out_C, stride = 1, kernel_size = config.MODEL_PARAMETERS['encoder_kernel_size'], padding = padding, bias = False)
                self.attentionlayer.weight.data = custom_filters
                self.attentionlayers.append(self.attentionlayer)

        self.img_branch = self.fcn(in_C = self.out_encoder_feats,
                                   feat_type = 'imag',
                                   clf_head = False)
        
        self.mixed_branch = self.fcn(in_C = (self.config.MODEL_PARAMETERS['imag_fcn'][-1] + self.config.NUM_CLINICAL_FEATURES),
                                     feat_type = 'mixed',
                                     clf_head = True)
        
        # self.img_clf_head = self.classification_head(in_C = self.config.MODEL_PARAMETERS['imag_fcn'][-1])
        
        # self.clin_branch = self.fcn(in_C = self.config.NUM_CLINICAL_FEATURES,
        #                             feat_type = 'clin',
        #                             clf_head = True)
        
            
    def forward(self, x_clin, x_img, x_msk):
        
        # Layer 1
        x_img = self.conv1(x_img)
        x_img = self.ds1(x_img)
        if(self.config.MODEL_PARAMETERS['attention_type'] == 1):
            x_msk = self.attentionlayers[0](x_msk)  
            x_msk = torch.where(x_msk > 0, 1, x_msk)
            x_msk = self.ds1(x_msk)
            x_img = x_img * x_msk
            # np.save('old_mask_1.npy', x_msk.cpu().detach().numpy())
            
        # Layer 2
        x_img = self.conv2(x_img)
        x_img = self.ds2(x_img)
        if(self.config.MODEL_PARAMETERS['attention_type'] == 1):
            x_msk = self.attentionlayers[1](x_msk)  
            x_msk = torch.where(x_msk > 0, 1, x_msk)
            x_msk = self.ds2(x_msk)
            x_img = x_img * x_msk
            
        # Layer 3
        x_img = self.conv3(x_img)
        x_img = self.ds3(x_img)
        if(self.config.MODEL_PARAMETERS['attention_type'] == 1):
            x_msk = self.attentionlayers[2](x_msk)  
            x_msk = torch.where(x_msk > 0, 1, x_msk)
            x_msk = self.ds3(x_msk)
            x_img = x_img * x_msk
        
        # Layer 4 (Optional)
        if len(self.config.MODEL_PARAMETERS['encoder_filters']) > 3:
            # Conv Block 4
            x_img = self.conv4(x_img)
            x_img = self.ds4(x_img)
            if(self.config.MODEL_PARAMETERS['attention_type'] == 1):
                x_msk = self.attentionlayers[3](x_msk)  
                x_msk = torch.where(x_msk > 0, 1, x_msk)
                x_msk = self.ds4(x_msk)
                x_img = x_img * x_msk
        
        # Flatten encoder output
        x_img = x_img.view(-1, self.out_encoder_feats)
        
        #####################################################################
        # Output
        x_out_1 = self.img_branch(x_img)
        x_out_2 = self.mixed_branch(torch.cat((x_out_1, x_clin), dim = 1))
        # x_out_3 = self.clin_branch(x_clin)
        # x_out_1 = self.img_clf_head(x_out_1)
        # return x_out_1, x_out_2, x_out_3
        return x_out_2
        #####################################################################
    
    
    def conv_layer(self, in_W, in_C, out_C):
        
        nn_seq_tot = []
        
        for i in range(self.config.MODEL_PARAMETERS['num_conv']):
            
            if i == 1: in_C = out_C
            
            nn_seq, out_W = self.conv_block(in_W = in_W,
                                            in_C = in_C,
                                            out_C = out_C,
                                            kernel_size = self.config.MODEL_PARAMETERS['encoder_kernel_size'],
                                            stride = 1)
            in_W = out_W
            nn_seq_tot.extend(nn_seq)
            
        return nn.Sequential(*nn_seq_tot), out_W
    
    
    def conv_block(self, in_W, in_C, out_C, kernel_size, stride):
        
        nn_seq = []
        
        # (1) Conv2D
        padding = (kernel_size - stride) // 2 # padding mode = "same"
        nn_seq.append(nn.Conv2d(in_channels = in_C,
                                out_channels = out_C,
                                kernel_size = kernel_size,
                                stride = stride,
                                padding = padding))
        out_W = (in_W - kernel_size + 2 * padding) // stride + 1 # out dim = [(W - K + 2P) // S] + 1
        
        # (2) Activation
        nn_seq.append(nn.LeakyReLU(negative_slope = 0.01, inplace = True))
        
        # (3) Normalization
        norm_type = self.config.MODEL_PARAMETERS['encoder_normalization']
        if (norm_type == 'batch'): # BatchNorm
            nn_seq.append(nn.BatchNorm2d(num_features = out_C))
        elif (norm_type == 'group'): # GroupNorm
            nn_seq.append(nn.GroupNorm(num_groups = self.config.MODEL_PARAMETERS['num_group_norm'],
                                       num_channels = out_C))
        elif (norm_type == 'instance'): # InstanceNorm
            nn_seq.append(nn.InstanceNorm2d(num_features = out_C))
        elif (norm_type == 'layer'): # LayerNorm
            nn_seq.append(nn.LayerNorm(normalized_shape = [out_C, out_W, out_W]))
        else:
            raise Exception("ERROR: Unknown normalization type || Please select one from ['batch', 'group', 'instance', 'layer']")
        
        # (4) DropOut
        nn_seq.append(nn.Dropout(self.config.MODEL_PARAMETERS['encoder_dropout_prob']))
        
        return nn_seq, out_W
    
    
    def ds_layer(self, in_W):
        
        nn_seq_tot = []
        
        kernel_sz = self.config.MODEL_PARAMETERS['encoder_kernel_size']
        ds_stride = self.config.MODEL_PARAMETERS['ds_stride']
        
        padding =  (kernel_sz - ds_stride) // 2 # padding mode = "same"
        nn_seq_tot.append(
            nn.MaxPool2d(kernel_size = kernel_sz,
                         padding = padding,
                         stride = ds_stride)
            )
        out_W = (in_W - kernel_sz + 2 * padding) // ds_stride + 1
        
        return nn.Sequential(*nn_seq_tot), out_W
    
    
    def fcn(self, in_C, feat_type, clf_head = False):
        
        nn_seq = []
        
        if (feat_type == 'clin'):
            fc_layers = self.config.MODEL_PARAMETERS['clin_fcn']
            drop_prob = self.config.MODEL_PARAMETERS["clin_dropout_prob"]
        elif (feat_type == 'imag'):
            fc_layers = self.config.MODEL_PARAMETERS['imag_fcn']
            drop_prob = self.config.MODEL_PARAMETERS['imag_dropout_prob']
        else:
            fc_layers = self.config.MODEL_PARAMETERS['mixed_fcn']
            drop_prob = self.config.MODEL_PARAMETERS['mixed_dropout_prob']
                
        for i in fc_layers:
            # (1) linear transformation
            nn_seq.append(nn.Linear(in_C, i))
            # (2) activation
            nn_seq.append(nn.LeakyReLU(negative_slope = 0.01, inplace = True))
            # (3) normalization
            nn_seq.append(nn.BatchNorm1d(i))
            # (4) regularization
            nn_seq.append(nn.Dropout(drop_prob))
            
            in_C = i
            
        if clf_head:
            nn_seq.append(nn.Linear(fc_layers[-1], self.config.MODEL_PARAMETERS['dfu_classes']))
            
        return nn.Sequential(*nn_seq)
    
    
    def classification_head(self, in_C):
        
        nn_seq = [nn.Linear(in_C, self.config.MODEL_PARAMETERS['dfu_classes'])]
        
        return nn.Sequential(*nn_seq)