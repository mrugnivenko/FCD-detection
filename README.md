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
