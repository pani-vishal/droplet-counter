# MLO-Project 2: Droplet Counter
## About
This repository contains the implementation and report for MLO Project 2 ([CS-433](https://www.epfl.ch/labs/mlo/machine-learning-cs-433/)). This project is in collaboration with the ([Laboratory of Biomedical Microfluidics (LBMM)] (https://www.epfl.ch/labs/lbmm/)) at EPFL and deals with counting number of aqueous droplets, in an emulsion, that contain bacteria. More information about the project and dataset is present in ***[project_description.pdf](./project_description.pdf)***.

## Setting up the project

The project uses anaconda to handle its environment. Procedure to install anaconda can be followed on [this](https://docs.anaconda.com/anaconda/install/index.html) link.

Once done, 

1. Move to this directory:
``` 
cd path/to/this/directory
``` 
2. Create a new environment with the following command:
``` 
conda env create --name <env_name> -f environment.yml 
```
3. Activate the environment using the command:
``` 
conda activate <env_name>
```

## Training the learning models
### The YOLOv5
[YOLOv5](https://github.com/ultralytics/yolov5) is used as the circle detector. To train the YOLO model, first we need to create a dataset in a format specific to YOLOv5.
#### Managing the dataset

1. The existing dataset can be found on this link. The next step can be skipped if the existing dataset is desired to be used.
2. To create the dataset from scratch:
   1. Keep all the training images in _datasets/droplets/train/original_.
   2. Keep all the testing images in _datasets/droplets/test/original_.
   3. Create 4 folders:
      1. _datasets/droplets/detector_ds/images/train_
      2. _datasets/droplets/detector_ds/images/test_
      3. _datasets/droplets/detector_ds/labels/train_
      4. _datasets/droplets/detector_ds/labels/test_
   4. Move to the following directory:
      ``` 
      cd src/data_utils/
      ``` 
   5. Run the following command
      ``` 
      python detector_ds_extractor.py
      ``` 

***Note: An example of the directory structure can be seen in the provided Google Drive link.***


#### Training
Once the dataset is ready, the detector can be trained.
1. Move to the YOLOv5 directory:
``` 
cd yolov5
``` 
2. Use the following command to train the model:
``` 
python train.py --img <img_size> --batch <batch_size> --epochs <num_epochs> --data <path_to_config_file> --weights <model_choice>
```
**Example**
```
python train.py --img 640 --batch 16 --epochs 3 --data ../configs/circle.yaml --weights yolov5s.pt
``` 

***Note: More information on training the YOLOv5 model can be found in the [YOLOv5 repository](https://github.com/ultralytics/yolov5/wiki/Train-Custom-Data).***


## Training the classifier
### Managing the dataset
### Training

## GUI
### Key bindings

## Libraries and packages used

# Made with ❤️ by:

1. Lovro Nuić, lovro.nuic@epfl.ch		
3. Vishal Pani, vishal.pani@epfl.ch
