# FCD-detection


## 3D-UNet on voxel-based data

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
