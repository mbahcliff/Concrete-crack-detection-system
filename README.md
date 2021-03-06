# Concrete-crack-detection-system
## Introduction
![3509673328_9b7ed82212_b-5705a84d3df78c7d9e953054](https://user-images.githubusercontent.com/63025220/110212643-64efc380-7e6a-11eb-9064-1a482b2737bc.jpg)

Concrete is the most widely used construction material in the entire world and arguably the most important material in civil engineering. Strength is one of the main reasons concrete has been used for constructions for many decades. It can easily withstand tensile and compressive stresses without getting affected. It is exceptionally durable and can last for ages as it can survive harsh weather conditions and disasters. It is rigid and resilient from deformation These characteristics, however, result in concrete structures lacking the flexibility to move in response to environmental or volume changes. Cracking is usually the first sign of distress in concrete [2]. It is, however, possible for deterioration to exist before cracks appear.

## Problem statement
Concrete surface cracks are major defect in civil structures (buildings, bridges, etc.) which if not treated on time, it would not only lead to the detrimental effect on its structural health and longevity but can cause real large-scale disasters which we have seen happened on many occasions in the past and claimed the lives of thousands. An example is the Dhaka garment factory collapse in Bangladesh. It occurred on 13th April 2013 and killed 1134 people and approximately 2500 injured [4]. To reduce such disasters from happening, structural inspection should be carried out on civil structures on regular basis. Structural inspection is done for the evaluation of rigidity and tensile strength of the structure. This is usually done by checking for cracks on concretes[2]. Crack detection plays a major role in a structural inspection process, finding the cracks and determining the building health. The cracked concretes are then replaced with new ones. As a data scientist I want to build a deep learning machine system that would be able to detect these crack concretes 

## Project objective
Creating a deep learning model using Convolutional neural network (CNN) algorithm that would alert you when a crack concrete is detected. Convolutional neural network is a class of deep neural networks, most applied to analyzing visual imagery. I will be making use of important CNN libraries such as keras.  Keras is an open-source software  library that provides a python interface for artificial neural networks. 

## Methodology
This is an image  classification problem which involves giving an image  as the input to a model built using a specific algorithm that outputs the class or the probability of the class that the image belongs to. This process in which we label an image to a particular class is called Supervised Learning. The algorithm is Convolutional neural network mostly used for visual imagery. CNN model consist of two main layers. The first part consists of the Convolutional layers and the Pooling layers in which the main feature extraction process takes place. In the second part, the Fully Connected and the Dense layers perform several non-linear transformations on the extracted features and act as the classifier part[3]. This model is explain and illustrated more detailly below
The metric I will be focusing on is a predicting metric. That is, a metrics better in predicting cracks 

## Related work
A  crack detection system was done with a similar dataset- structural network defect dataset(SDNET2018)  which contains over 56000  images of cracked and non-cracked concrete bridge decks, walls, and pavements. Most of the images were captured from Utah State University. The system has an accuracy of 76%

Link: https://www.kaggle.com/kanikepratap/cnn-assignment-ii-submitted.

Milind Raj worked on a similar data using RESNET50 and obtained an accuracy rate of 95.3%

Link: https://github.com/MILIND-RAJ/Concrete-Crack-Images-Classification-Using-ResNet50/blob/master/Final_Assignment%20concrete.ipynb. 

## Business goal
For the projects which I have come across which used a similar dataset as seen above, an accuracy in the range of 76- 95.3% was obtained. This project would be successful if I'm able to obtained an accuracy of at least 99% or better.

## Dataset
### Description
The data set is from the Mendeley data. The data is collected from various Middle East Technical University (METU) Campus Buildings. The dataset was published on 23rd June 2019 by Çağlar Fırat Özgenel [1]
My dataset is unstructured. The dataset is divided into two as negative and positive crack images for image classification. Each class has 20000images with a total of 40000 images with 227 x 227 pixels with RGB channels.

Link to dataset website: https://data.mendeley.com/datasets/5y9wdsg2zt/2

Link to dataset: https://md-datasets-cache-zipfiles-prod.s3.eu-west-1.amazonaws.com/5y9wdsg2zt-2.zip

## Data splitting, preprocessing and generators

Using train_test_split from  Sklearn.model_selection library , my data is split into train and test data in the ratio 80:20

This ratio gives us a total of 32000 images reserve for training and 8000 images for testing.

Normalization is the most crucial step in image pre-processing. This refers to rescaling the pixel values so that they lie within a confined range. Our original images consist in RGB coefficients in the 0-255 , but such values would be too high for this model to process , so we target values between 0  and 1 instead by  scaling with a 1/255.

Create three generators train_gen, validation_gen and test_gen each with validated images belonging to two classes

In the prepocessing phase, validation_split is o0.1 of my train dataset

These classes are either positive or negative. The three generators are:

Train_gen:  28800 validated image filenames in two classes

Validated_gen: 3200 validated image fienames in two classes

Test_gen: 8000 validated image filenames in two classes

## Data Visualization

This inolves understanding my datasets by placing it in a visual context so that patterns, trends and correlations that might not otherwise be detected can be exposed.

### label count

![Capture8](https://user-images.githubusercontent.com/63025220/110213452-1a704600-7e6e-11eb-95e1-de0a50b596f2.PNG)

### Preview of images in train and test dataset

![Capture11](https://user-images.githubusercontent.com/63025220/110213557-966a8e00-7e6e-11eb-8061-d8575870e2a7.PNG)

![Capture13](https://user-images.githubusercontent.com/63025220/110213585-a84c3100-7e6e-11eb-9103-1171b31c0863.PNG)

## Data augmentation
Image Augmentation is a way of applying different types of transformation techniques on actual images, thus producing copies of the same image with alterations. This helps to train deep learning models on more image variations than what is present in the actual dataset 

