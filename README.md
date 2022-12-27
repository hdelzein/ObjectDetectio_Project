# Object Detection in an Urban Environment

## Project Overview

This is SSD (Single Shot Detector) model training project. The goal is to classify objects of cars, pedestrians and 
cyclists as a classfication problem and to determine the location of these objects or their bounding boxs in the images
as a regression problem. 

For this project, we will be using data from the [Waymo Open dataset](https://waymo.com/open/) and a pretrained model 
which is the ssd_resnet50_v1_fpn_640x640_coco17_tpu-8.tar.gz. We shall use Tensor flow object detetction API to custimize
the training of the model. The Tensor Object Detection API relies on a config file to custimize the model training.

## project Setup

For this project I used the Udacity Desktop ennviroment fot training and evaluation. The structure of 
the data is as follow:

### Data Structure 
The data you will use for training, validation and testing is organized as follow:
```
/home/workspace/data/waymo
    - training_and_validation - contains 97 files to train and validate your models
    - train: contain the train data (empty to start)
    - val: contain the val data (empty to start)
    - test - contains 3 files to test your model and create inference videos
```
### Experiments Folder

This is the default experiments folder structure provided by Udacity:
```
experiments/
    - pretrained_model/
    - exporter_main_v2.py - to create an inference model
    - model_main_tf2.py - to launch training
    - reference/ - reference training with the unchanged config file
    - experiment0/ - create a new folder for each experiment you run
    - label_map.pbtxt
    ...
```
I created a Final_Config_File folder in GitHub that include the final version of the pipeline_new.config
file that I used to train and evaluate the final model.

```
Final_Config_File/
    - pretrained_model/pipeline_new.config

```


## Data Set

### Data Exploratory Analysis 

<img src="DataImages/one_one.png" width="1000">
<img src="DataImages/two_one.png" width="1000">
<img src="DataImages/three_one.png" width="1000">
<img src="DataImages/four_one.png" width="1000">
<img src="DataImages/five_one.png" width="1000">
<img src="DataImages/six_one.png" width="1000">
<img src="DataImages/seven_one.png" width="1000">
<img src="DataImages/eight_one.png" width="1000">
<img src="DataImages/nine_one.png" width="1000">
<img src="DataImages/ten_one.png" width="1000">

### Data Augmentation Analysis

<img src="augImages/flipHorizantal.png" width="150" title = "Horizantol Flip"> <img src="augImages/cropImage.png" width="150" title = "Crop Image">
<img src="augImages/adjustBright.png" width="150" title = "Adjust Brightness"> <img src="augImages/adjustContrast.png" width="150" title = "Adjust Contrast">

## Training

### Training Experiment


<img src="TensorBoardFigures/totalLoss.jpg" width="250"/> <img src="TensorBoardFigures/learning rate.jpg" width="250"/>
<img src="TensorBoardFigures/class_local_loss.jpg" width="500">
<img src="TensorBoardFigures/norm_reg_loss.jpg" width="500">
<img src="TensorBoardFigures/detectPrecisionOne.jpg" width="500">
<img src="TensorBoardFigures/detectPrecisionTwo.jpg" width="500">
<img src="TensorBoardFigures/detectPrecisionThree.jpg" width="500">
<img src="TensorBoardFigures/recallOne.jpg" width="500">
<img src="TensorBoardFigures/recallTwo.jpg" width="500">
<img src="TensorBoardFigures/recallThree.jpg" width="500">


### Improve the reference

How I run it to improve performace
