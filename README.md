# Concrete-crack-detection-system
## Introduction
Concrete is the most widely used construction material in the entire world and arguably the most important material in civil engineering. Strength is one of the main reasons concrete has been used for constructions for many decades. It can easily withstand tensile and compressive stresses without getting affected. It is exceptionally durable and can last for ages as it can survive harsh weather conditions and disasters. It is rigid and resilient from deformation These characteristics, however, result in concrete structures lacking the flexibility to move in response to environmental or volume changes. Cracking is usually the first sign of distress in concrete [1]. It is, however, possible for deterioration to exist before cracks appear.
## Problem statement
Concrete surface cracks are major defect in civil structures (buildings, bridges, etc.) which if not treated on time, it would not only lead to the detrimental effect on its structural health and longevity but can cause real large-scale disasters which we have seen happened on many occasions in the past and claimed the lives of thousands. An example is the Dhaka garment factory collapse in Bangladesh. It occurred on 13th April 2013 and killed 1134 people and approximately 2500 injured [2]. To reduce such disasters from happening, structural inspection should be carried out on civil structures on regular basis. Structural inspection is done for the evaluation of rigidity and tensile strength of the structure. This is usually done by checking for cracks on concretes[3]. Crack detection plays a major role in a structural inspection process, finding the cracks and determining the building health. The cracked concretes are then replaced with new ones.
## Project objective
Creating a deep learning model using convolutional neural network and  keras that would alert you when a cracked concrete is detected. Keras is an open-source software library that provides a Python interface for artificial neural networks.
## Methodology
I will using Covolutional neural network to build a system that would be able to predict whether a concrate is cracked or not.My project includes the following steps outline below;
Loading, exploration and visualization of image data

Data augmentation

Convolutional neural network model

Model training

Model Evaluation

Accuracy and loss graphs

Predictions 

Classification report

Result
## Business goal
Given that the highest accuracy rate which I know  of so far for such a project is about 99.6%, this project would be successful if I'm able to obtained an accuracy of at least 99.7% accuracy or better
## Dataset
### Description
The data set is from the Mendeley data. The data is collected from various Middle East Technical University (METU) Campus Buildings. The dataset was published on 23rd June 2019 by Çağlar Fırat Özgenel [4]
My dataset is unstructured. The dataset is divided into two as negative and positive crack images for image classification. Each class has 20000images with a total of 40000 images with 227 x 227 pixels with RGB channels.

Link to dataset website: https://data.mendeley.com/datasets/5y9wdsg2zt/2

Link to dataset: https://md-datasets-cache-zipfiles-prod.s3.eu-west-1.amazonaws.com/5y9wdsg2zt-2.zip
## Related works

