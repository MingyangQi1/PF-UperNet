# PF-UperNet
# Method for Segmentation of Bean Crop and Weeds Based on Improved UperNet--Code
#### This experiment is based on the mmsegmentation project. The installation environment is based on the files of this project. To reproduce the experiment, simply match the files in this repository to the corresponding file paths in the mmsegmentation project.Additionally, you need to install the DCNv2 environment. fpn_poolformer.py is a configuration file, placed in the config folder. poolformer.py needs to be placed in mmcls for reference. uper_head.py file is placed in the mmseg folder. This repository contains all the innovative points of the experiment. For details, please see the paper：Method for Segmentation of Bean Crop and Weeds Based on Improved UperNet.
# Data
#### The original dataset is from https://doi.org/10.15454/JMKP9S . Due to the large size of the dataset, it cannot be uploaded. Please download it yourself. The dataset consists of 300 sets of image data, each of which contains an original image and a label image. In this experiment, the dataset was divided into training, test, and validation sets in a ratio of 7:2:1. The training set of this paper used a total of 9 offline data augmentation methods.
# Results
#### Use the trained model to do segmentation on test images, the result is statisfactory.
