**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./model.png "Model Visualization"
[image2]: ./before_cropping.jpg "Example image before cropping"
[image3]: ./after_cropping.jpg "Example image after cropping"
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

The model for this project built consists of a convolution neural network with 5x5 and 3x3 filter sizes and depths between 6 and 32 (model.py lines 66, 68, 70 and 72). At every convolutional layer, a pooling layer is appended to it. 

The model includes RELU layers to introduce nonlinearity, and the data is normalized in the model using a Keras lambda layer (code line 62). Another layer performs image cropping to remove the landscape and the car body from the image (code line 63).

Three dense layers with a decreasing number of hidden units are added (model.py lines 75, 77 and 79), along with dropout layers in order to reduce overfitting (model.py lines 76 and 78). 

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 49-52). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track. This is shown later in the video.

#### 1. Model parameter tuning

The model used the *RMSprop* optimizer predefined in Keras, so the learning rate was not tuned manually (model.py line 90).

#### 2. Appropriate training data

Initially, data had been collected by driving the car in the simulator two laps in the forward direction and two more laps in the opposite direction. The problem with the model was the car couldn’t properly learn the features associated with the bridge section. At the end the data set provided by Udacity was used to train the model.

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to use similar networks built in previous projects since there wasn't any template to build the model from...

My first attempt was to apply tranfer learning to the Inception v3 model. But the model was able to predict a single steering angle leading to a horrible driving performance. That strategy was dimissed in favor of a convolution neural network model with few convolutional layers and few dense layers. Subsequent iterations of the model helped to decrease the training loss. I thought this model might be appropriate because it was easily debuggable.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting. 

A key point has been to find a good balance between the number of epochs and the batch size. These factors are strongly influenced by your computer hardware.

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track when approaching to the bridge or while driving through it... to improve the driving behavior in these cases, I made the following improvements:

* To increase the number of hidden units in the last dense layer (from 16 to 32).
* To add a fourth convolutional layer.
* To decrease the correction on the steering angle from 1 to 0.2.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road. The validation error obtained was around 0.014.

Here is a visualization of the network architecture 

![alt text][image1]

The output layer minimizes the mean square error (MSE) of the  real and predicted steering angle of the car.

#### 2. Creation of the Training Set & Training Process

Images from Udacity were good enough to train the model. The 20% of all the images where randomly assigned to the test set and the rest to train the model. Besides the usual normalization process and cropping the images no other method was used in the preprocessing step.

![alt text][image2]  ![alt text][image3]

The image on the left shows an image from the front camera of the car and the image on the right shows the output image from the cropping layer.
