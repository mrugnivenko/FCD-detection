# FCD-detection

## Description of the problem and obtained results

Focal cortical dysplasia (FCD) is one of the most common epileptogenic lesions associated with cortical development malformations. However, the accurate detection of the FCD relies on the radiologist professionalism, and in many cases, the lesion could be missed. We’ve developed architectures for solving the problem of automatic detection of FCD on 3D MR images using 2 independent approaches: 3D UNet on voxel-based data and ResNet on point clouds. Both architectures were not previously used for that task. Before solving the real problem of FCD detection we first tried to solve the toy ones: grey matter and hippocampus segmentation. Both architectures proved their reasonable applicability on toy problems, but the results on the real problem were not so satisfying. Absence of high quality on the real problem could be explained by lack of data. Experiment results of point cloud-based ResNet showed more accurate results and seem to be more promising than results of voxel-based 3D UNet. As we’ve faced the lack of the data problem, the only further development is to collect more data and rerun the experiments. 


## 3D-UNet on voxel-based data

### Requirements
* numpy 1.19.0
* pandas 1.2.1
* tqdm 4.19.9
* nibabel 2.4.0
* scikit-learn 0.24.1
* torch 1.4.0
* unet 0.7.3
* matplotlib 3.3.3
* IPython 7.9.0
* comet_ml 3.2.1
* torchio 0.18.23

### File Structure
* landmarks  - directory with landmarks for Histogram Standardization
* targets - directory with .csv file, which consists all the available inforamtion about targets
* utils - directory with all the tools used in project

   data_processor.py - processing data for single-headed 3D UNet
   
   data_processor_two_head.py - processing data for two-headed 3D UNet
   
   metrics.py and metrics_deep_mind.py - metrics used for quality assessment

   routine.py - architecture of single-headed 3D UNet and its training

   routine_two_head.py - architecture of two-headed 3D UNet and its training

   visualization_tools.py - functions for simple visualization of MR data and predictions
   
* weights - saved weights of 3D UNet for different experiments
* T1_to_seg.ipynb - training for transfer-learining
* T1_or_seg_or_T2_to_fcd.ipynb - FCD detection fine-tuning with single-headed net
* T1+seg_to_fcd.ipynb - FCD detection fine-tuning with two-headed net 

### Usage


## ResNet on Point Clouds
### Requirements
* Ubuntu 16.04
* Anaconda with python=3.6
* pytorch>=1.3
* torchvision with pillow<7
* cuda=10.1
### File Structure
* cfgs/brain  - directory with configurations for models in each experiment
* datasets, function, ops, utils  - directories with all the tools used in the project
* models  - directory with models used in the project
* Brain_to_point_clouds.ipynb - transforms 3d tensors of brains into Point Clouds and saves it
* 1 experiment.ipynb - training the model to detect hippocampus with pointcloud of brain without intensities
* 2 experiment.ipynb - training the model to detect hippocampus with pointcloud of brain with intensities
* 3 experiment.ipynb - training the model to detect FCD with pointcloud of brain without intensities
* 4 experiment.ipynb - training the model to detect FCD with pointcloud of brain with intensities
* visualisation N exp.ipynb (N = 1,2,3,4) - visualisation of losses (train, test) and main metrics (test) for the aformentioned experiments.
### Usage
#### Point clouds generation
To convert MR images into the point clouds for experiments 1-4 one needs to first put them in .nii format into the datasets/fcd_classification_bank/ folder. Then run the Brain_to_point_clouds.ipynb notebook.
#### Training
To conduct any particular experiment out of 4, one needs to first complete "Point clouds generation" step, then training and validation process can be strated with "N experiment.ipynb" (N = 1,2,3,4). Experiment can be conducted in any order. 
#### Visualisation
After the experiments you may want to visualise loss functions and see the corresponding metrics. In order to do that you have to run "visualisation N exp.ipynb" (N = 1,2,3,4).
