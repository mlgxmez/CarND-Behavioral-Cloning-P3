**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/placeholder.png "Model Visualization"
[image2]: ./examples/placeholder.png "Grayscaling"
[image3]: ./examples/placeholder_small.png "Recovery Image"
[image4]: ./examples/placeholder_small.png "Recovery Image"
[image5]: ./examples/placeholder_small.png "Recovery Image"
[image6]: ./examples/placeholder_small.png "Normal Image"
[image7]: ./examples/placeholder_small.png "Flipped Image"

## Rubric Points

### Files Submitted & Code Quality

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md or writeup_report.pdf summarizing the results
* run0.mp4 a video recording of the car driving around the first track in autonomous mode

Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

The model for this project built consists of a convolution neural network with 5x5 and 3x3 filter sizes and depths between 6 and 12 (model.py lines 65, 67, 69 and 71). At every convolutional layer, a pooling layer is appended to it. 

The model includes RELU layers to introduce nonlinearity, and the data is normalized in the model using a Keras lambda layer (code line 61). Another layer performs image cropping to remove the landscape and the car body from the image (code line 62).

Three dense layers with a decreasing number of hidden units are added (model.py lines 74, 76 and 78), along with dropout layers in order to reduce overfitting (model.py lines 75 and 77). 

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 49-52). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track. This is shown later in the video.

#### 1. Model parameter tuning

The model used the rmsprop optimizer predefined in Keras, so the learning rate was not tuned manually (model.py line 87).

#### 2. Appropriate training data

Initially, data had been collected by driving the car in the simulator two laps in the forward direction and two more laps in the opposite direction. The problem with the model was the car couldn’t properly learn the features associated with the bridge section. At the end the data set provided by Udacity was used to train the model.

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to use similar networks built in previous projects since there wasn't any template to build the model from. ..

My first attempt was to apply tranfer learning to the Inception v3 model. But the model was able to predict a single steering angle leading to a horrible driving performance. That strategy was dimissed in favor of a convolution neural network model with few convolutional layers and few dense layers. Subsequent iterations of the model helped to decrease the training loss. I thought this model might be appropriate because it was easily debuggable.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting. 

A key point has been to find a good balance between the number of epochs and the batch size. These factors are strongly influenced by your computer hardware.

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track when approaching to the bridge or while driving through it... to improve the driving behavior in these cases, I increased the number of hidden units in the last dense layer and added a fourth convolutional layer.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

Here is a visualization of the network architecture 

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 160x320x3 RGB image   							| 
| Lambda            |                   |
| Cropping          | outputs 90x320x3  |
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 88x318x6 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 44x159x6 				|
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 42x157x6 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 20x77x6 				|
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 18x75x6 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 8x36x6 				|
| Convolution 3x3     	| 1x1 stride, valid padding, outputs 8x36x12 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 3x7x12 				|
| Flatten  |  outputs 612      |
| Dense		| inputs 612, outputs 512    						|
| RELU					|												|
| Dropout					|												|
| Dense	| inputs 512, outputs 256    						|
| RELU					|												|
| Dropout					|												|
| Dense		| inputs 256, outputs 32   						|
| Dense 	| inputs 32, outputs 1   						|

<!--![alt text][image1]-->

The output layer minimizes the mean square error (MSE) of the  real and predicted steering angle of the car.

#### 2. Creation of the Training Set & Training Process

Images from Udacity were good enough to train the model. The 20% of all the images where randomly assigned to the test set and the rest to train the model. Besides the usual normalization process and cropping the images no other method was used in the preprocessing step.
