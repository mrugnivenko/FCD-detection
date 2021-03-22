'''
Description of the following fucntions:
    * get_torchio_dataset
    * get_loaders
    * get_model_and_optimizer
    * prepare_batch
    * forward
    * run_epoch
    * train

Code adapted from: https://github.com/fepegar/torchio#credits

    Credit: Pérez-García et al., 2020, TorchIO: 
    a Python library for efficient loading, preprocessing, 
    augmentation and patch-based sampling of medical images in deep learning.
'''

import torchio
from torchio import AFFINE, DATA, PATH, TYPE, STEM
from torchio.transforms import (
    RandomFlip, RandomAffine,
    RandomElasticDeformation,
    RandomNoise, RandomMotion,
    RandomBiasField, RescaleIntensity,
    Resample, ToCanonical,
    ZNormalization, CropOrPad,
    HistogramStandardization,
    OneOf, Compose,
)

import torch
import torch.nn as nn 
import torch.nn.functional as F
from unet import UNet

import time
import enum
import warnings
import numpy as np
import multiprocessing
import matplotlib.pyplot as plt
from tqdm import tqdm_notebook, tqdm
from IPython.display import clear_output

import imp 
import utils.metrics as metrics
imp.reload(metrics)
from utils.metrics import *

device = torch.device('cuda') if torch.cuda.is_available() else 'cpu'

# WHAT???
LIST_FCD =  [ 8,   10,   11,   12,   13,    16,   17,   18,  26,  47, 49,   50, 
  51,   52,   53,   54,   58,  85,  251,  252,  253,  254,  255]

# We work with torch.tensor with 5 dimension, so its shape e.g (1,1,200,200,200)
BATCH_DIMENSION = 0
CHANNELS_DIMENSION = 1
SPATIAL_DIMENSIONS = 2, 3, 4


def get_torchio_dataset(inputs, predictions, targets, transform):
    
    """
    Function creates a torchio.SubjectsDataset from inputs and targets lists and applies transform to that dataset
    
    Arguments:
        * inputs (list): list of paths to MR images
        * targets (list):  list of paths to ground truth segmentation of MR images
        * transform (False/torchio.transforms): transformations which will be applied to MR images and ground truth segmentation of MR images (but not all of them)
    
    Output:
        * datasets (torchio.SubjectsDataset): it's kind of torchio list of torchio.data.subject.Subject entities
    """
    
    subjects = []
    for (image_path, pred_path, label_path) in zip(inputs, predictions, targets ):
        subject_dict = {
            'MRI' : torchio.Image(image_path, torchio.INTENSITY),
            'PRED': torchio.Image(pred_path, torchio.LABEL),
            'LABEL': torchio.Image(label_path, torchio.LABEL), #intensity transformations won't be applied to torchio.LABEL 
        }
        subject = torchio.Subject(subject_dict)
        subjects.append(subject)
    
    if transform:
        dataset = torchio.SubjectsDataset(subjects, transform = transform)
    elif not transform:
        dataset = torchio.SubjectsDataset(subjects)
    
    return dataset

def get_loaders(data, cv_split, training_transform = False,
        validation_transform = False, training_batch_size = 1,
        validation_batch_size = 1):
    
    """
    Function creates dataloaders 
    
    Arguments:
        * data (data_processor.DataMriSegmentation): torchio dataset
        * cv_split (list): list of two arrays, one with train indexes, other with test indexes
        * training_transform (bool/torchio.transforms): whether or not to use transform for training images
        * validation_transform (bool/torchio.transforms): whether or not to use  transform for validation images
        * patch_size (int): size of patches
        * patches (bool): if True, than patch-based training will be applied
        https://niftynet.readthedocs.io/en/dev/window_sizes.html - about patch based training
        * samples_per_volume (int): number of patches to extract from each volume
        * max_queue_length (int): maximum number of patches that can be stored in the queue
        * training_batch_size (int): size of batches for training
        * validation_batch_size (int): size of batches for validation
        * mask (bool): if True, than masked images will be used 
    
    Output:
        * training_loader (torch.utils.data.DataLoader): loader for train
        * validation_loader (torch.utils.data.DataLoader): loader for test
    """
    
    training_idx, validation_idx = cv_split
    mask = data.mask
    
    print('Training set:', len(training_idx), 'subjects')
    print('Validation set:', len(validation_idx), 'subjects')
    
    training_set = get_torchio_dataset(
        list(data.img_files[training_idx].values), 
        list(data.img_pred[training_idx].values), 
        list(data.img_seg[training_idx].values),
        training_transform)

    validation_set = get_torchio_dataset(
        list(data.img_files[validation_idx].values),
        list(data.img_pred[validation_idx].values),
        list(data.img_seg[validation_idx].values),
        validation_transform)
    
    if mask in ['bb', 'combined']:
            print(f'Mask type is {mask}')
            # if using masked data for training
            training_set = get_torchio_dataset(
                list(data.img_files[training_idx].values),
                list(data.img_pred[training_idx].values), 
                list(data.img_mask[training_idx].values),
                training_transform)

            validation_set = get_torchio_dataset(
                list(data.img_files[validation_idx].values),
                list(data.img_pred[training_idx].values), 
                list(data.img_mask[validation_idx].values),
                validation_transform)
    
    training_loader = torch.utils.data.DataLoader(
            training_set, batch_size = training_batch_size)

    validation_loader = torch.utils.data.DataLoader(
            validation_set, batch_size = validation_batch_size)
        
            
    print('Training loader length:', len(training_loader))
    print('Validation loader length:', len(validation_loader))
    
    return training_loader, validation_loader

def get_model_and_optimizer(device, 
                            num_encoding_blocks = 3,
                            out_channels_first_layer = 16,
                            patience = 3):
    
    '''
    Function creates model, optimizer and scheduler
    
    Arguments:
     * device (cpu or gpu): device on which calculation will be done 
     * num_encoding_blocks (int): number of encoding blocks, which consist of con3d + ReLU + conv3d + ReLu
     * out_channels_first_layer (int) : number of channels after first encoding block
     * patience (int): Number of epochs with no improvement after which learning rate will be reduced.
     
    Output:
     * model 
     * optimizer
     * scheduler
    '''
    
    # reproducibility
    # https://pytorch.org/docs/stable/notes/randomness.html
    torch.manual_seed(0)
    np.random.seed(0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    #https://segmentation-models.readthedocs.io/en/latest/tutorial.html
    model = UNet(
        in_channels = 2,
        out_classes = 2,
        dimensions = 3,
        num_encoding_blocks = num_encoding_blocks,
        out_channels_first_layer = out_channels_first_layer,
        normalization = 'batch',
        upsampling_type = 'linear',
        padding = True,
        activation = 'PReLU',
    ).to(device)
    
    optimizer = torch.optim.AdamW(model.parameters())
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                           mode = 'min',
                                                           factor = 0.01,
                                                           patience = patience,
                                                           threshold = 0.01)
    
    return model, optimizer, scheduler

def prepare_batch(batch, device, task = 'T1_seg_to_fcd'):
    
    """
    WORK ONLY WITH BATCH SIZE = 1, I WILL IMPROVE IT LATER
    Function sends MRI data to the devise. Binarizes LABEL, creates tensor of shape (1, 2, X, Y, Z) for further calculation of the dice score and
    also sends it ti the devise
    
    Arguments:
        * batch (dict): batch dict, contains input data and target data
        * device (torch.device): device for computation 
    
    Output:
        * inputs (torch.tensor): inputs in the appropriate format for model 
        * targets (torch.tensor): targets in the appropriate format for model 
    """
    if task == 'T1+pred_of_seg_to_seg':
        inputs1 = batch['MRI'][DATA]
        inputs2 = batch['PRED'][DATA].float()
        inputs = torch.cat([inputs1, inputs2], dim=1).to(device)
        
        targets = batch['LABEL'][DATA]
        targets[0][0][(np.isin(targets[0][0], LIST_FCD))] = 1 
        targets[targets >= 900] = 1.0 #GET RID OF CONCSTANT
        targets[targets != 1] = 0.0
        targets_2_dim = torch.stack((targets[0][0] , 1 - targets[0][0])) #WORKS ONLY IF BATCH_SIZE = 1
        targets_2_dim = targets_2_dim.unsqueeze(0)
        targets_2_dim = targets_2_dim.to(device)   
        
    if task == 'T1_seg_to_fcd':
        inputs1 = batch['MRI'][DATA]
        inputs2 = batch['PRED'][DATA].float()
        inputs2[0][0][(np.isin(inputs2[0][0], LIST_FCD))] = 1 
        inputs2[inputs2 >= 900] = 1.0 #GET RID OF CONCSTANT
        inputs2[inputs2 != 1] = 0.0
        inputs = torch.cat([inputs1, inputs2], dim=1).to(device)
        
        targets = batch['LABEL'][DATA]
        targets_2_dim = torch.stack((targets[0][0] , 1 - targets[0][0])) #WORKS ONLY IF BATCH_SIZE = 1
        targets_2_dim = targets_2_dim.unsqueeze(0)
        targets_2_dim = targets_2_dim.to(device)   

    return inputs, targets_2_dim

def forward(model, input_):
    '''
    Function apply model to the input
    
    Arguments:
        * model
        * input_ (torch.tensor): (N, 1, X, Y, Z) batch with MRI data
    
    Output:
        * logits (torch.tensor): (1, 2, X, Y, Z) probabilities tensor, one component 
        is probability-tensor (1,X,Y,Z) to be the brain, another component 
        is probability-tensor (1,X,Y,Z) to be background. 
        In general the shape of the tensor is (N, 2, X, Y, Z), where N is batch size
    '''
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning)
        logits = model(input_)
    return logits

class Action(enum.Enum):
    '''
    Class used for understanding whether model has to perform training stage or testing
    '''
    
    TRAIN = 'Training'
    VALIDATE = 'Validation'

def run_epoch(epoch_idx, action, loader, model, optimizer, ratio, scheduler = False, experiment = False, loss_type = False, task = 'T1_to_seg'):
    
    '''
    Function runs one epoch with set parameters
    
    Arguments:
        * epoch_idx (int): number of epoch
        * action (Action.TRAIN or Action.VALIDATE): indicator whether model has to perform training stage or testing
        * experiment (): experiment tool from Comet.ml
    
    Output:
        * epoch_losses (np.array): list of losses of every batch
    '''

    is_training = (action == Action.TRAIN)
    model.train(is_training) #Sets the module in training mode if is_training = True
    
    epoch_losses = []

    for batch_idx, batch in enumerate(tqdm(loader)):
        inputs, targets = prepare_batch(batch, device, task)
        inputs.shape
        targets = targets.float()
        inputs = inputs.float()
        optimizer.zero_grad()
        
        with torch.set_grad_enabled(is_training):
            if loss_type == 'dice':
                logits = forward(model, inputs)
                probabilities = F.softmax(logits, dim = CHANNELS_DIMENSION)
                batch_losses = get_dice_loss(probabilities, targets)
                batch_loss = (batch_losses*torch.tensor([1, 0]).float().to(device)).sum()
            
            if loss_type == 'dice+ce':
                logits = forward(model, inputs)
                probabilities = F.softmax(logits, dim = CHANNELS_DIMENSION)
                batch_losses = get_dice_loss(probabilities, targets)
                batch_loss = (batch_losses*torch.tensor([1, 0]).float().to(device)).sum()
                
                ce_loss_func = nn.BCELoss()
                ce_loss = ce_loss_func(probabilities, targets.detach())
                batch_loss = batch_loss + ce_loss
                
            if loss_type == 'weighted ce': 
                ce_loss_func = nn.BCELoss()
                logits = forward(model, inputs)
                probabilities = F.softmax(logits, dim = CHANNELS_DIMENSION)
                
                weights = targets*ratio + (1-targets)
                class_weights = weights.float().to(device)
                w_ce_loss_func = nn.BCELoss(weight = class_weights)
                w_ce_loss = w_ce_loss_func(probabilities, targets.detach())
                batch_losses = w_ce_loss
                batch_loss = batch_losses
                
            if loss_type == 'ce in target class': 
                ce_loss_func = nn.BCELoss()
                logits = forward(model, inputs)
                probabilities = F.softmax(logits, dim = CHANNELS_DIMENSION)
                
                weights = targets
                class_weights = weights.float().to(device)
                w_ce_loss_func = nn.BCELoss(weight = class_weights)
                w_ce_loss = w_ce_loss_func(probabilities, targets.detach())
                batch_losses = w_ce_loss
                batch_loss = batch_losses
                
            if is_training:
                batch_loss.backward()
                optimizer.step()
        
            epoch_losses.append(batch_loss.item())
           
            if experiment:
                if action == Action.TRAIN:
                    experiment.log_metric("train_dice_loss", np.array(epoch_losses).mean())
                elif action == Action.VALIDATE:
                    experiment.log_metric("validate_dice_loss", np.array(epoch_losses).mean())
                    
            del inputs, targets, logits, probabilities, batch_losses
 
    epoch_losses = np.array(epoch_losses)
    
    return epoch_losses 


def train(num_epochs, training_loader, validation_loader, model, optimizer, ratio, scheduler, 
          weights_stem, save_epoch= 1, experiment= False, verbose = True, loss_type = False, task = 'T1_seg_to_fcd'):
    
    '''
    Fucntion trains model with set parameters
    
    Arguments:
        * num_epochs (int): number of epochs for training
        * weights_stem (str): name for experiment logs
        * save_epoch (int): how often save weights of model
        * experiment (experiment variable ot False): name of Comet.ml variable for experiment
        * verbose (bool): whether to print and plot results after epochs or not 
    '''
    
    start_time = time.time()
    epoch_train_loss, epoch_val_loss = [], []
    

    run_epoch(0, Action.VALIDATE, validation_loader, model, optimizer, ratio, scheduler, experiment, loss_type, task)
    
    for epoch_idx in range(1, num_epochs + 1):
        

        epoch_train_losses = run_epoch(epoch_idx, Action.TRAIN, training_loader, 
                                       model, optimizer, ratio, scheduler, experiment, loss_type, task)
  
        epoch_val_losses = run_epoch(epoch_idx, Action.VALIDATE, validation_loader, 
                                     model, optimizer, ratio, scheduler, experiment, loss_type, task)
        
        # 4. Print metrics
        if verbose:
            clear_output(True)
            print("Epoch {} of {} took {:.3f}s".format(epoch_idx, num_epochs, time.time() - start_time))
            print("  training loss (in-iteration): \t{:.6f}".format(epoch_train_losses[-1]))
            print("  validation loss: \t\t\t{:.6f}".format(epoch_val_losses[-1]))    
        
        epoch_train_loss.append(np.mean(epoch_train_losses))
        epoch_val_loss.append(np.mean(epoch_val_losses))
        
        # 5. Plot metrics
        if verbose:
            plt.figure(figsize=(10, 5))
            plt.plot(epoch_train_loss, label='train')
            plt.plot(epoch_val_loss, label='val')
            plt.xlabel('epoch')
            plt.ylabel('loss')
            plt.legend()
            plt.show()
        
        if scheduler:     
            scheduler.step(np.mean(epoch_val_losses))
        if experiment:
            experiment.log_epoch_end(epoch_idx)
        if (epoch_idx% save_epoch == 0):
            torch.save(model.state_dict(), f'weights/{weights_stem}_epoch_{epoch_idx}.pth')  