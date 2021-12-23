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

1. The existing dataset can be found on this [link](https://drive.google.com/drive/folders/16Y5xFiwxop042F-vzUEGk-LMmsV9sEGf?usp=sharing). The next step can be skipped if the existing dataset is desired to be used.
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

***Note: An example of the directory structure can be seen in the provided [Google Drive link](https://drive.google.com/drive/folders/16Y5xFiwxop042F-vzUEGk-LMmsV9sEGf?usp=sharing).***


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
Droplet classifier with options for VGG16, VGG19, and Resnet50 are provided.
### Managing the dataset
1. The existing dataset can be found on this [link](https://drive.google.com/drive/folders/16Y5xFiwxop042F-vzUEGk-LMmsV9sEGf?usp=sharing). The next step can be skipped if the existing dataset is desired to be used.
2. Make appropriate folders as mentioned in ```input_folders``` and ```output_folders``` in [pd_data_extraction.ipynb](src/data_utils/pd_data_extraction.ipynb).
3. Generate the required images by running [pd_data_extraction.ipynb](src/data_utils/pd_data_extraction.ipynb).
4. Upload all of these images to a labeling site. In project an online platform [Label Studio](https://labelstud.io/) was used.
5. Label empty droplets as "Empty", droplets with bacteria as "Bacteria", and circles which are not droplets as "Not_Droplet".
6. Import the label.csv.
### Training
To train the classification models run [train_models.ipynb](./train_models.ipynb), until the Inference section. 

## GUI
The GUI provides an intuitive way to do inference and annotate the images. To run the GUI, type the following command:
```
python gui.py
``` 

```diff
!Note: Ensure that the environment is activated before running any of the above commands
``` 
### Annotated window
![](report/images/gui_annotated.png)
### Key bindings
| Command | Description |
| --- | --- |
| `Mouse Click + Drag` | When done on circles, it changes their location |
| `Mouse Scroll Up/Down` | Increase/Decrease size of selected circle |
| `Numpad +/-` | Zoom In/Out of the image |
| `R` | Remove selected circle |
| `N` | Create a new circle |
| `C` | Clear all circle ***(With great power, comes great responsibility!!)*** |
| `1` | Change class of selected droplet to _Bacteria_ |
| `2` | Change class of selected droplet to _Empty_ |
| `3` | Change class of selected droplet to _Not_Droplet_ |

## Extra Remarks
The baseline classifier using histogram analysis can be found and run from [baseline.ipynb](./ntbks/baseline.ipynb)
## External libraries and packages used
1. [OpenCV](https://opencv.org/)
2. [PyTorch](https://pytorch.org/)
3. [Tensorflow](https://www.tensorflow.org/)
4. [Keras](https://keras.io/)
5. [Scikit-Learn](https://scikit-learn.org/stable/)

# Made with ❤️ by:

1. Lovro Nuić, lovro.nuic@epfl.ch		
3. Vishal Pani, vishal.pani@epfl.ch
