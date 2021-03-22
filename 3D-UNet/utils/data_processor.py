'''
Description of the following functions:
    * load_nii_to_array
    * crope_image
    * get_targets_info
    
Also description of class DataMriSegmentation
'''

import os
import glob
import numpy as np
import pandas as pd
from tqdm import tqdm
import nibabel as nib
pd.options.mode.chained_assignment = None 
from sklearn.preprocessing import LabelEncoder

import torch
import torch.utils.data as data

LIST_FCD =  [ 8,   10,   11,   12,   13,    16,   17,   18,  26, 31, 47, 49,   50, 
  51,   52,   53,   54,   58,  85,  251,  252,  253,  254,  255]


def load_nii_to_array(nii_path):
    
    """ 
    Function returns np.array data from the *.nii file
    
    Arguments:
        * nii_path (str): path to *.nii file with data

    Output:
        * data (np.array): data obtained from nii_path
    """
    
    try:
        data = np.asanyarray(nib.load(nii_path).dataobj)
        return (data)
    
    except OSError:
        print(FileNotFoundError(f'No such file or no access: {nii_path}'))
        return('')

def crope_image(img, coord_min, img_shape):
    
    """
    Function cropes an image 
    
    Arguments:
        * img (np.array): MR image
        * coord_min (x_min, y_min, z_min):  It will be (0, 0, 0) point in cropped image
        * img_shape (x_shape, y_shape, z_shape): desired image shape

    Output:
        * img (np.array): croped image with size (1, x_shape, y_shape, z_shape)
    """
    
    img = img[coord_min[0]:coord_min[0] + img_shape[0],
              coord_min[1]:coord_min[1] + img_shape[1],
              coord_min[2]:coord_min[2] + img_shape[2],]
    
    if img.shape[:3] != img_shape:
        print(f'Current image shape: {img.shape[:3]}')
        print(f'Desired image shape: {img_shape}')
        raise AssertionError
        
    return img.reshape((1,) + img_shape)

def get_targets_info(sample, 
                     targets_path = 'targets/targets_fcd_bank.csv',
                     image_path='../../datasets/fcd_classification_bank',
                     mask_path = False,
                     prefix = False,
                     data_type = False,
                     ignore_missing = True):
    
    '''
    Function to obtain DataFrame with all the information about
    needed MR images (including paths to MRI and to file with ground truth segmentation).
    Walks through directories and completes DataFrame, according to targets.
    
    Arguments:
        * targets_path (str): path to DataFrame with all the information about MR images
        * sample (str): name of the medical center with images from which we want to work, 'all' for whole centers
        * prefix (str): patient name prefix (optional). E.g 'fcd' means taht we want to work only with fcd patients
        * mask_path (str or False): paths to folder with masks
        * image_path (str): path to nii.gz files with MR data
        * data_type (str or False): str = {'img', 'seg'}. If e.g data_type = 'img' - only MR image is used for model.
        'img' or 'seg' are using in classification, in segmentation data_type = False, since we want to have both 
        * ignore_missing (bool): whether we want to remove examples with missing data or not
    
    Outputs:
        * files: DataFrame with all the information about targets, which are needed for our task
        (including paths to MRI and to file with ground truth segmentation)
    '''
    
    description_of_targets = pd.read_csv(targets_path, index_col = 0)
    '''
    DataFrame with infromation about each MRI
        * sample: 'pirogov', 'la5_study', 'soloviev', 'hcp', 'kulakov'. Names of medical centers from which MR images were obtained
        * patien: names of MR images 
        * fcd: 1 or 0. Responsible for the presence or absence of the FCD in MR images  
        * age: ages of patients, whose MR images we have
        * gender: gender of patients, whose MR images we have
        * scan: . To be honest don't know 
        * detection: ['mri_positive', 'mri_negative', nan, 'mri_positive/mri_negative']. Whether doctors found FCD using MRI or not
        * comments: [nan,  0.,  3.,  2.,  1.]. To be honest don't know
    '''
    
    files = pd.DataFrame(columns = ['patient','scan','fcd','img_file','img_seg']) # output DataFrame
    condition_about_sample = (description_of_targets['sample'] == sample)
                         
    if prefix:
        condition_about_sample = (description_of_targets['sample'] == sample)&(description_of_targets['patient'].str.startswith(prefix))
        
    files['patient'] = description_of_targets['patient'][condition_about_sample].copy()
    files['fcd'] = description_of_targets['fcd'][condition_about_sample].copy()
    files['scan'] = description_of_targets['scan'][condition_about_sample].copy()
    files['detection'] = description_of_targets['detection'][condition_about_sample].copy()
    files['comments'] = description_of_targets['comments'][condition_about_sample].copy()
    
    if mask_path:
        files['img_mask'] = ''
    
    elif sample == 'all':
        files['patient'] = description_of_targets['patient'].copy()
        files['fcd'] = description_of_targets['fcd'].copy()
        files['scan'] = description_of_targets['scan'].copy()
        files['detection'] = description_of_targets['detection'].copy()
        files['comments'] = description_of_targets['comments'].copy()
        
    # paths to MRI and corresponding segmentation are adding
    for i in tqdm(range(len(files))):
        for file_in_folder in glob.glob(os.path.join(image_path,'*norm*')):
            if sample == 'pirogov':
                if (files['patient'].iloc[i] + '_norm.nii.gz') == file_in_folder.split('/')[-1]:
                    files['img_file'].iloc[i] = file_in_folder
            else:
                if files['patient'].iloc[i] in file_in_folder:
                    files['img_file'].iloc[i] = file_in_folder

        for file_in_folder in glob.glob(os.path.join(image_path,'*aseg*')):
            if sample == 'pirogov':
                if ((files['patient'].iloc[i] +'_aparc+aseg.nii.gz') == file_in_folder.split('/')[-1]) or\
                ((files['patient'].iloc[i] +'_aparc+aseg.nii') == file_in_folder.split('/')[-1]):
                    files['img_seg'].iloc[i] = file_in_folder 
            else:    
                if files['patient'].iloc[i] in file_in_folder:
                    files['img_seg'].iloc[i] = file_in_folder   

        if mask_path:
            for file_in_folder in glob.glob(os.path.join(mask_path,'*.nii*')):
                if (files['patient'].iloc[i] +'.nii.gz') == file_in_folder.split('/')[-1]:
                    files['img_mask'].iloc[i] = file_in_folder 
    
    # treating missing data
    if ignore_missing:
        # if only 'img' is needed for classification
        if data_type =='img':
            files.dropna(subset = ['img_file'], inplace= True)
        # if only 'seg' is needed for classification
        elif data_type =='seg':
            files.dropna(subset = ['img_seg'], inplace= True)
        # saving only full pairs of data. Mandatory for segmentation 
        else: 
            files.dropna(subset = ['img_seg','img_file'], inplace= True)

    # reindexing an array, since we droped NaNs.
    files = files.reset_index(drop=True)
    label_encoder = LabelEncoder() 
    files['scan'] = label_encoder.fit_transform(files['scan'])

    return files, label_encoder

class DataMriSegmentation(data.Dataset):
    
    """
    Arguments:
        image_path (str): path to nii.gz files with MR data  
        mask_path (str): paths to folder with masks 
        prefix (str): patient name prefix (optional). E.g 'fcd' means taht we want to work only with fcd patients
        sample (str): name of the medical center with images from which we want to work, 'all' for whole centers
        targets_path (str): path to DataFrame with all the information about MR images
        ignore_missing (bool): whether we want to remove examples with missing data or not
        mask (string): ['seg', 'bb', 'combined']. Type of mask to use in task   
    """
    
    def __init__(self, sample,
                 image_path = '../../datasets/fcd_classification_bank',
                 targets_path = 'targets/targets_fcd_bank.csv',
                 mask_path = False,
                 prefix = False,
                 ignore_missing = True,
                 coord_min = (30,30,30, ),
                 img_shape = (192, 192, 192, ),
                 mask = 'seg'):
        
        super(DataMriSegmentation, self).__init__()
        
        print(f'Assembling data for: {sample} sample.')
        files, label_encoder = get_targets_info(sample, targets_path, image_path,
                                 mask_path, prefix, ignore_missing)
        
        self.img_files = files['img_file']
        self.img_seg = files['img_seg']
        self.scan = files['scan']
        self.scan_keys = label_encoder.classes_
        self.target = files['fcd'] 
        self.detection = files['detection']
        self.misc = files['comments']
        
        if mask_path:
            self.img_mask = files['img_mask']
            
        self.coord_min = coord_min
        self.img_shape = img_shape
        self.mask_path = mask_path
        self.mask = mask
        
        assert mask in ['seg','bb','combined'], "Invalid mask name!"
            
    def __getitem__(self, index):
        img_path = self.img_files[index]
        seg_path = self.img_seg[index]

        img_array = load_nii_to_array(img_path)
        seg_array = load_nii_to_array(seg_path)

        img = crope_image(img_array, self.coord_min, self.img_shape)
        seg = crope_image(seg_array, self.coord_min, self.img_shape)

        if self.mask == 'seg':
            # binarising cortical structures
            # GET RID OF CONSTANT
            seg[np.isin(seg, LIST_FCD)] = 1.
            seg[seg >= 900] = 1
            seg[seg != 1.] = 0.
            return torch.from_numpy(img).float(), torch.from_numpy(seg).float()

        elif self.mask == 'bb':
            # preparing bounding box mask 
            bb_mask_path = self.img_mask[index]
            mask_array = load_nii_to_array(bb_mask_path)
            masked_img = crope_image(mask_array, self.coord_min, self.img_shape)
            return torch.from_numpy(img).float(), torch.from_numpy(masked_img).float()

        elif self.mask == 'combined':
            # binarising cortical structures
            # GET RID OF CONSTANT
            seg[np.isin(seg, LIST_FCD)] = 1.
            seg[seg >= 900] = 1
            seg[seg != 1.] = 0.

            # preparing bounding box mask 
            bb_mask_path = self.img_mask[index]
            mask_array = load_nii_to_array(bb_mask_path)
            masked_img = crope_image(mask_array, self.coord_min, self.img_shape)

            # calculating combined mask as intersection of both masks
            comb_mask = np.logical_and(masked_img, seg)
            return torch.from_numpy(img).float(), torch.from_numpy(comb_mask).float()

    def __len__(self):
        return len(self.img_files)

def get_ration_of_ones(data, mask = 'seg'):
    
    '''
    Function gives ratio of ones in whole data sample to define weighted CE loss 
    
    Arguments: 
        * data (DataMriSegmentation): data of DataMriSegmentation class
        * mask (str): type of mask
        
    Inputs:
        out(int): the ratio of ones in image and mask
    '''
    
    ones = 0 
    if mask == 'seg':
        for i in range(len(data)):
            img, seg = data[i]
            seg[0, :][np.isin(seg[0, :], LIST_FCD)] = 1.
            seg[0, :][seg[0, :] >= 900] = 1
            seg[0, :][seg[0, :] != 1.] = 0.
            ones += seg[0,:].sum()
            
    if mask == 'combined':
        for i in range(len(data)):
            img, seg = data[i]
            seg[0, :][np.isin(seg[0, :], LIST_FCD)] = 1.
            seg[0, :][seg[0, :] >= 900] = 1
            seg[0, :][seg[0, :] != 1.] = 0.

            # preparing bounding box mask 
            bb_mask_path = data.img_mask[i]
            mask_array = load_nii_to_array(bb_mask_path)
            masked_img = crope_image(mask_array, data.coord_min, data.img_shape)
            
            # calculating combined mask as intersection of both masks
            comb_mask = np.logical_and(masked_img, seg[0, :])
            
            ones += comb_mask[0,:].sum()

    return int(len(data)*seg.shape[1]*seg.shape[2]*seg.shape[3]/ones.numpy())


def create_combined_masks(data, path = '../datasets/ellipse_masks/combined/'):
    
    '''
    Function creates combined masks
    
    Argumetns:
        * data (DataMriSegmentation): 
        * path (str): path to save masks
    '''
    
    for index in range(len(data)):
        img_path = data.img_files[index]
        seg_path = data.img_seg[index]
        seg_array = load_nii_to_array(seg_path)
        seg = seg_array

        seg[np.isin(seg, LIST_FCD)] = 1.
        seg[seg >= 900] = 1.
        seg[seg != 1.] = 0.

        # preparing bounding box mask 
        bb_mask_path = data.img_mask[index]
        mask_array = load_nii_to_array(bb_mask_path)
        masked_img = mask_array
        
        img1 = nib.load(img_path)
        affine = img1.affine
        # calculating combined mask as intersection of both masks
        comb_mask = np.logical_and(masked_img, seg)
        new_image = nib.Nifti1Image(comb_mask.astype(float), affine)
        nib.save(new_image, path+img_path.split('/')[-1].split('_norm')[0]+'.nii.gz')