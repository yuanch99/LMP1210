This directory contatins the Mask_R-CNN config files and data 

mmdetection: 
https://github.com/open-mmlab/mmdetection

All data used  can be found in the google drive link here
 - SN: 
https://drive.google.com/drive/folders/1oeQIl40KVICoPKB8u8XZ8cibeo9-QUDk?usp=sharing
 - CWM: 
https://drive.google.com/drive/folders/19Xo3YOi4GYO706w2J7SFPJBMDvORGEuT?usp=sharing

Both annotations files and image folders need to be saved

Link to the colab (runtime has to be GPU): 
https://colab.research.google.com/drive/1YG1_cnWyVJNSYB0r8__OwlkarzwnAi_x?usp=sharing

Download the configs and create 2 folders under mmdetection called SN and CWM. Add the specific configs to these folders for model testing and training. 

Single image inferencing configs are already specified. 

General workflow: 

Building the model and the dataset
- First, we need to point to the config file of our model to our dataset
- Specify each of the train, test, and validation datasets (images and COCO annotations files) 
- Specify the hyperparameters

Training

Inference

Author: Peter Her
