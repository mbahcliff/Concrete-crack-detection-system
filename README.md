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

### Preview of images in train dataset

![Capture11](https://user-images.githubusercontent.com/63025220/110213557-966a8e00-7e6e-11eb-8061-d8575870e2a7.PNG)

### Preview of images in test dataset

![Capture13](https://user-images.githubusercontent.com/63025220/110213585-a84c3100-7e6e-11eb-9103-1171b31c0863.PNG)

## Data augmentation
Image Augmentation is a way of applying different types of transformation techniques on actual images, thus producing copies of the same image with alterations. This helps to train deep learning models on more image variations than what is present in the actual datasetRandom shift
### Random rotation
The image rotation technique enables the model by generating images of different orientations. The ImageDataGenerator class in Keras uses this technique to generate randomly rotated images in which the angle can range from 0 degrees to 360 degrees. 

![Captur1](https://user-images.githubusercontent.com/63025220/110213921-1e9d6300-7e70-11eb-834c-6da4d89cce80.PNG)

### Random shift

The random shifts technique helps in improving those images that are not properly positioned . Keras ImageDataGenerator uses parameters height_shift_range for vertical shifts in an image and for horizontal shifts in an image, we can use width_shift_range

### Vertical random shift

![Captur2](https://user-images.githubusercontent.com/63025220/110214085-f2cead00-7e70-11eb-8822-7ee82ce9ff03.PNG)

### Horizontal random shift

![Captur3](https://user-images.githubusercontent.com/63025220/110214171-6d97c800-7e71-11eb-8fa3-bc4c07a351f8.PNG)

Other augmentation techniques ellaborated in the notebook inludes random flips, random brightness and random zoom

## Building the CNN model
### Creating convolutional base
The model has  different layers:

Input layer:  This is a layer in which the input image is fed into the CNN model. The input is of shape (image_height, image_width, color_channels).  For this project my input shape is (120, 120, 3)

Three convolutional layer each followed by a MaxPooling ;

Convolutional layer consist of a filter, kernel_size and activation function. This is the stage in which most of the base features such as sharp edges and curves are extracted from the image and hence this layer is also known as the feature extractor layer. 

Pooling layer: The pooling operation is also known as down sampling where the spatial volume of the image is reduced. If we perform a Pooling operation with a stride of 2 on an image with dimensions 28×28, then the image size reduced to 14×14, it gets reduced to half of its original size. 

The activation is a mathematical gate in between the input feeding the current neuron and its output  going to the next layer. They basically decide whether the neuron should be activated or not. 

ReLU activation function is widely used and is default choice as it yields better result

Convolutional base summary

![Captur11](https://user-images.githubusercontent.com/63025220/110214628-74273f00-7e73-11eb-8361-de3610b816ff.PNG)

### Adding the dense layer on top

To complete the model, feed the last output tensor from the convolutional base (of shape (26, 26, 32)) into one or more Dense layers to perform classification. Dense layers take vectors as input (which are 1D), while the current output is a 3D tensor. First, flatten (or unroll) the 3D output to 1D, then add one or more Dense layers on top. My image dataset has 2 output classes, so I will use a final Dense layer with 2 outputs. 

Model summary

![Captur12](https://user-images.githubusercontent.com/63025220/110214719-e9930f80-7e73-11eb-9c51-fc19fe47bbbd.PNG)

## Compiling and training the model

After the model is created, it is compiled using the Adam optimizer, one of  the most popular optimization algorithm. Additionally, the loss type is specified as  categorical cross entropy which is used for multi class classification, binary cross entropy can also be used as the loss function. Lastly, the metrics is specified as accuracy

### Compiling the model

To compile our model, the following parameters are needed: Optimizers, loss function and metrics
### Optimizer

Adam optimizers perform the best on average  compared  to other optimizers.

### Loss function

The loss function used to compile our model is categorical cross entropy. It is used as a loss function for multi class classification model. That is when we have two or more output labels
### Metrics

The metrics is accuracy

### Training the model

The model is trained with keras fit() function. The model trains for 15 epochs. The fit() function will return a history object; By storing the result of the function in a history, it is used to plot the accuracy and the loss function plots between the training and validation which will help to visualize our model's performance visually. 

![Capture51](https://user-images.githubusercontent.com/63025220/117546158-13dd8680-aff7-11eb-8c2f-35b05a2843b3.PNG)


To train our model, the following parameters are needed: number of epochs, validation data and train data

### Epoch

An epoch represents one training step

From the figure above, after 15 epochs a train accuracy of 99.57% was obtained

## Model evaluation on test accuracy

### Accuracy:  99.55% ,  Loss:  2.47%

An accuracy of 99.55% looks impressive!

## Visualizing the  accuracy and loss of my model

Putting my model evaluation into perspective by plotting  the accuracy  and loss plots of the training and validation data

![Capture50](https://user-images.githubusercontent.com/63025220/117546133-fa3c3f00-aff6-11eb-9300-1d8a70b1e1ed.PNG)

## Predicting Labels

Model test data prediction gives us floating point values.  It will not be feasible to compare the predicted labels with true test labels. So, I will round off the output which will convert the float values into integers. Further more, I will use np.argmax() to select the index number which has a higher value in a row
Numpy argmax () is an inbuilt function that is used to get the indices of the maximum element from our array (single dimension array) or any row or column of any given array

![Capture57](https://user-images.githubusercontent.com/63025220/117545762-2787ed80-aff5-11eb-90dd-6b0ce890169c.PNG)

![Capture52](https://user-images.githubusercontent.com/63025220/117545783-45555280-aff5-11eb-9a10-abad84426f4f.PNG)

## Confusion matrix

A confusion matrix is a predictive analytics tool. Specifically, it is a table that displays and compares actual values with the model’s predicted values .

![Capture55](https://user-images.githubusercontent.com/63025220/117545705-e1328e80-aff4-11eb-887d-f6d17884395c.PNG)

### True Positive (TP)

True positive represents the value of correct predictions of positives out of actual positive cases. Out of 4003 actual positives, 3991 are correctly predicted positive. Thus, the value of true positive is 3991.

### False Positive (FP)

It represents the value of incorrect positive predictions. The value represents the number of negatives(out of 3997) which gets falsely predicted a positive. Out of 3997 actual negatives, 6 is falsely predicted as positive. Thus the value of false positive is 6.

### True Negative (TN)

True negative represents the value of correct prediction of negatives out of actual negative cases. Out of 3997 actual negatives , 3991 are corrected predicted as negatives. The value of true negatives is 3991.

### False Negative (FN)

False Negative represents the value of incorrect negative predictions. This value represents the  number of actual positives (out of 4003) which gets falsely predicted as negatives. Out of 4003 actual positives, 12 is incorrectly predicted as negatives. Thus the value of False Negative is 12

## Performance metrics

### Precision

It represents the model's ability to correctly predict the positives out of all the positive prediction it made. It represents the ratio between the number positive samples correctly classified to the total number of samples classified as positive (either correctly or incorrectly).
### Precision score =  TP/(TP+FP)
###                 = (3991/(6+3978)
###                 =  1.00

### Recall

Model recall score represents the model's ability to correctly predict the positives out of actual positives. The recall is calculated as the ratio of the true positives to the actual positives
### Recall score = TP/(TP+FN)
###              = 3991/(3991+12)
###              = 1.00

### Accuracy score

It represents the model's ability to correctly predict both the positives and negatives out of all the predictions.
### Accuracy score = (TP+TN)/(TP+FP+TN+FN)
###                =  (3991+3991) / (3991 + 12 + 3991 + 6)
###                =    1.00

### F1-Score

It represents the model's score as a function of precision  and recall score. The F1-score is a way of combining the precision and recall of the model, It is also known as the harmonic mean of the model's precision and  recall
### F1-score  = 2 X Precision score X Recall score / (Precision score + Recall score)
###           =  (2 X 1.00x1.00) /( 1.00+ 1.00)
###           = 1.00

## Classification report

Classification report gives us a summary table containing precision, recall, F1-score and makes it easy for us to observe which class performs better

![Capture56](https://user-images.githubusercontent.com/63025220/117545984-2905e580-aff6-11eb-894c-9ee3c5ee3f6b.PNG)

## Pros and cons of the CNN model

### Pros
In terms of performance, CNN model is very efficient relative to other models

It is simple to implement and requires fewer parameters to build

It is used in various fields and perform major tasks like facial recognition, analyzing documents, understanding climate, image recognition and object identification

### Cons

Overfitting is a common problem when training the model especially when we don’t have enough data to train our model with. This problem  can be resolve by increasing the data through augmentation.

Another common issue is data loss during the training process. This can be reduced by increasing the number of epochs and increasing the amount of data through data augmentation.

If the CNN has several layers, then the training process takes a lot of time if the computer doesn’t consist of a good GPU.

## Conclusion

Both classes performed the same in terms of precision, recall and f1-score

My model has a test accuracy of 99.721% which is better than my initial goal of 99.000%. Therefore, my project is successful


## Link to phasee 0 video: https://www.youtube.com/watch?v=CMisxnjuEik

## Link to phase 1 video: https://www.youtube.com/watch?v=djxdvKlHlhA

## Link to phase 2 video: https://www.youtube.com/watch?v=kYlO-trq0Bw

## References

[1] Özgenel, Ç. F. (2019, 07 23). Mendeley Data. Retrieved from Concrete Crack Images for Classification: https://data.mendeley.com/datasets/5y9wdsg2zt/2

[2] SCIENTIFIC, G. (2019, August 17). Evaluating Cracking in Concrete: Procedures. Retrieved from GIATEC: https://www.giatecscientific.com/education/cracking-in-concrete-procedures/

[3] Vadapalli, P. (2021, February 25). Image Classification in CNN: Everything You Need to Know. Retrieved from upgradeblog: https://www.upgrad.com/blog/image-classification-in-cnn/

[4] Wikipedia. (2021, February 13). 2013 Dhaka garment factory collapse. Retrieved from Wikipedia: https://en.wikipedia.org/wiki/2013_Dhaka_garment_factory_collapse

[5] GreekForGreek. (2020, May 18). Keras.Conv2D Class. Retrieved from GreekForGreek: https://www.geeksforgeeks.org/keras-conv2d-class/

[6] Lathiya, A. (9, September 5). Numpy Argmax: How To Use Np Argmax() Function In Python. Retrieved from https://appdividend.com/2020/03/28/python-numpy-argmax-function-example/

[7] ScienceDirect. (2021). Convolutional Layer. Retrieved from ScienceDirect : https://www.sciencedirect.com/topics/engineering/convolutional-layer
