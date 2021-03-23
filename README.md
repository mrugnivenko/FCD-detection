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
* cfgs/brain  - directory with configurations for models in each experiment
* datasets, function, ops, utils  - directories with all the tools used in the project
* models  - directory with models used in the project
* Brain_to_point_clouds.ipynb - transforms 3d tensors of brains into Point Clouds and saves it
* 1 experiment.ipynb - training the model to detect hippocampus with pointcloud of brain without intensities
* 2 experiment.ipynb - training the model to detect hippocampus with pointcloud of brain with intensities
* 3 experiment.ipynb - training the model to detect FCD with pointcloud of brain without intensities
* 4 experiment.ipynb - training the model to detect FCD with pointcloud of brain with intensities
* visualisation N exp.ipynb (N = 1,2,3,4) - visualisation of losses (train, test) and main metrics (test) for the aformentioned experiments.
