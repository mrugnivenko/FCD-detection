'''
Description of the following fucntions:
    * get_dice_score
    * get_dice_loss
'''

import torch
import numpy as np

def get_dice_score(output, target, SPATIAL_DIMENSIONS = (2, 3, 4), epsilon = 1e-9):
    '''
    Function gets dice score on output and target tensors  
    https://www.jeremyjordan.me/semantic-segmentation/#loss
    
    Arguments:
        * output (torch.tensor): (1,2,X,Y,Z) probabilities tensor, one component 
        is probability-tensor (1,X,Y,Z) to be the brain, another component 
        is probability-tensor (1,X,Y,Z) to be background. 
        In general the shape of the tensor is (N, 2, X, Y, Z), where N is batch size
        * target (torch.tensor): (1,2,X,Y,Z)  binary tensor, one component 
        is binary-mask (1,X,Y,Z) for the brain, another component 
        is binary-mask (1,X,Y,Z) for the background.
        In general the shape of the tensor is (N, 2, X, Y, Z), where N is batch size
        * SPATIAL_DIMENSIONS (typle): typle with indexes corresponding to spatial parts of tensors
        * epsilon (float): a small number used for numerical stability to avoid divide by zero errors  
    
    Outputs:
        * dice score (torch.tensor): tensor with dice score for every class on the image 
    '''
    
    p0 = output
    g0 = target
    p1 = 1 - p0
    g1 = 1 - g0
    
    tp = (p0 * g0).sum(dim=SPATIAL_DIMENSIONS)
    fp = (p0 * g1).sum(dim=SPATIAL_DIMENSIONS)
    fn = (p1 * g0).sum(dim=SPATIAL_DIMENSIONS)
    
    num = 2 * tp
    denom = 2 * tp + fp + fn
    
    dice_score = (num + epsilon)/(denom + epsilon)
    
    return dice_score


def get_dice_loss(output, target):
    '''
    Function gets dice score loss on output and target tensors  
    
    Arguments:
        * output (torch.tensor):  (1,2,X,Y,Z) probabilities tensor, one component 
        is probability-tensor (1,X,Y,Z) to be the brain, another component 
        is probability-tensor (1,X,Y,Z) to be background.
        In general the shape of the tensor is (N, 2, X, Y, Z), where N is batch size
        * target (torch.tensor): (1,2,X,Y,Z) binary tensor, one component 
        is binary-mask (1,X,Y,Z) for the brain, another component 
        is binary-mask (1,X,Y,Z) for the background.
        In general the shape of the tensor is (N, 2, X, Y, Z), where N is batch size
    
    Outputs:
        * dice score loss (torch.tensor): tensor with dice score loss for every class on the image 
    '''
    return 1 - get_dice_score(output, target)

def get_iou_score(prediction, ground_truth):
    '''
    Fucntion computes IoU of prediction of target and ground truth target
    
    Arguments:
        * prediction (np.array): predicted segmentation (with or without mask, whatever)
        * ground_truth (np.array):ground truth segmentation
    
    Outputs:
        * iou_score (float): IoU score 
    
    '''
    intersection, union = 0, 0
    intersection += np.logical_and(prediction > 0, ground_truth > 0).astype(np.float32).sum() 
    union += np.logical_or(prediction > 0, ground_truth > 0).astype(np.float32).sum()
    iou_score = float(intersection) / union
    return iou_score