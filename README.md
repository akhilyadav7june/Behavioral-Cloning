# **Behavioral Cloning** 

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/ModelSummary.jpg "Model Summary"
[image2]: ./examples/left_center_right.jpg "Left center right camera images"
[image3]: ./examples/Recovery_from_left_to_center.jpg "Recovery from left to center image"
[image4]: ./examples/Recovery_from_right_to_center.jpg "Recovery from right to center image"
[image5]: ./examples/original_flipped_image.jpg "Original and Flipped image"


## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.html summarizing the results
* run73.mp4 video recording of 1 lap

#### 2. Submission includes functional code

The code is written on model.py. When training finish on the model the result saved in model.h5. The model model.h5 is required as a parameter when car need to execute in autonomously.
Using the Udacity provided simulator and drive.py file and saved model model.h5, the car can be driven autonomously around the track by executing.
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the model after trained. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

I used the network which is used by NVIDIA for autonomous car driving. 

My final model consisted of the following layers:

* 3@160x320 Input with lambda layer for noralization
* Cropping layer by 70 pixel from above and 25 pixel from bottom of the image
* 3@5x5 convolution filter with strides 2x2 and relu activation
* 24@5x5 convolution filter with strides 2x2 and relu activation
* 36@5x5 convolution filter with strides 2x2 and relu activation
* 48@5x5 convolution filter with relu activation
* 64@3x3 convolution filter with relu activation
* 64@3x3 convolution filter with relu activation
* Flatten layer
* 100 outputs fully connected layer
* Dropout layer
* 50 outputs fully connected layer
* Dropout layer
* 10 outputs fully connected layer
* Dropout layer
* 1 output layer.

The model includes RELU layers to introduce nonlinearity (model.py lines 75 to 79), and the data is normalized in the model using a Keras lambda layer (model.py line 69). 

#### 2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting (model.py lines 82, 84 and 86). 

The model was trained and validated on different data sets to ensure that the model was not overfitting.
For that i used existing sample data and recorded 1 more lap and more data where track takes turn. Also i flipped the data and also used left and right images to get better generalization on the track.
The model is tested by running it through the simulator and vehicle is moving on the track pretty well.

#### 3. Model parameter tuning

I used following parameter:

    epochs : 10
    optimizer : Adam optimizer
    
The model used an adam optimizer, so the learning rate was not tuned manually. (model.py line 92).

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road with different angle. Split the whole data in training and validation. 205 used for validation only.  

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

My first step was to use a convolution neural network model similar to the Lenet architecture with existing sample data. I thought this model might be appropriate because it is good model for traffic sign image recognition.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set(20 % of whole data). And shuffling the data by passing the shuffle=True parameter in model.fit method. I executed the model and got the low mean squared on both training and validation set. 

``
Epoch 1/5
6428/6428 [==============================] - 12s 2ms/step - loss: 1.1607 - val_loss: 0.0156
Epoch 2/5
6428/6428 [==============================] - 9s 1ms/step - loss: 0.0132 - val_loss: 0.0136
Epoch 3/5
6428/6428 [==============================] - 9s 1ms/step - loss: 0.0113 - val_loss: 0.0126
Epoch 4/5
6428/6428 [==============================] - 9s 1ms/step - loss: 0.0100 - val_loss: 0.0121
Epoch 5/5
6428/6428 [==============================] - 10s 2ms/step - loss: 0.0090 - val_loss: 0.0115
``

Even though the loss is less but car is not driving properly on the track.

Then i used augmentation by flipping all the images. And used left and right camera image for training with little 0.2 correction factor in steering angle. Also cropped the hills and tree areas from the images. 70 pixels from up and 25 pixel from bottom of the image to cut uneccesary details. Again eexcuted many time with different no. of epochs. The model is better now but still the same issue. Not good at turns.
And some time go off the tracks.

``
Epoch 1/5
38572/38572 [==============================] - 47s 1ms/step - loss: 0.0263 - val_loss: 0.0180
Epoch 2/5
38572/38572 [==============================] - 43s 1ms/step - loss: 0.0152 - val_loss: 0.0171
Epoch 3/5
38572/38572 [==============================] - 43s 1ms/step - loss: 0.0142 - val_loss: 0.0195
Epoch 4/5
38572/38572 [==============================] - 43s 1ms/step - loss: 0.0140 - val_loss: 0.0180
Epoch 4/5
38572/38572 [==============================] - 43s 1ms/step - loss: 0.0135 - val_loss: 0.0170
``

After a lot change when the model is not learning i decide to use the NVIDIA model which is already tested. NVIDIA also used this model for real car driving on the roads.

I found that my model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting. 

``
Epoch 1/5
38572/38572 [==============================] - 48s 1ms/step - loss: 0.0173 - val_loss: 0.0646
Epoch 2/5
38572/38572 [==============================] - 47s 1ms/step - loss: 0.0147 - val_loss: 0.0723
Epoch 3/5
38572/38572 [==============================] - 43s 1ms/step - loss: 0.0134 - val_loss: 0.0647
Epoch 4/5
38572/38572 [==============================] - 43s 1ms/step - loss: 0.0124 - val_loss: 0.0677
Epoch 5/5
38572/38572 [==============================] - 43s 1ms/step - loss: 0.0115 - val_loss: 0.0665
``

To combat the overfitting, first i recorded more data specially the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to get back to center. Also recorded the data where road is turning. And then i used  Dropout(0.5).
With epochs 10 when model is trained i got below loss.

``
57369/57369 [==============================] - 75s 1ms/step - loss: 0.0244 - val_loss: 0.0398
Epoch 2/10
57369/57369 [==============================] - 70s 1ms/step - loss: 0.0213 - val_loss: 0.0387
Epoch 3/10
57369/57369 [==============================] - 70s 1ms/step - loss: 0.0201 - val_loss: 0.0378
Epoch 4/10
57369/57369 [==============================] - 70s 1ms/step - loss: 0.0194 - val_loss: 0.0387
Epoch 5/10
57369/57369 [==============================] - 70s 1ms/step - loss: 0.0187 - val_loss: 0.0382
Epoch 6/10
57369/57369 [==============================] - 70s 1ms/step - loss: 0.0181 - val_loss: 0.0377
Epoch 7/10
57369/57369 [==============================] - 70s 1ms/step - loss: 0.0179 - val_loss: 0.0378
Epoch 8/10
57369/57369 [==============================] - 70s 1ms/step - loss: 0.0173 - val_loss: 0.0383
Epoch 9/10
57369/57369 [==============================] - 70s 1ms/step - loss: 0.0170 - val_loss: 0.0380
Epoch 10/10
57369/57369 [==============================] - 70s 1ms/step - loss: 0.0168 - val_loss: 0.0383
``

Finally when i ran my trained model for track1 in simulator vehicle is driving autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture (model.py lines 66-98) consisted of a convolution neural network with the following layers and layer sizes:

Summary of the model lookes like below:

This summary contains information about the shape of the layers and the number of trainable parameters.

![alt text][image1]

#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I used existing sample data and recorded 1 more lap and also recorded the areas where more turns on the track. Here is an example of images taken from left, center and right camera.

![alt text][image2]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to drive when it drive to left or right side. Then it back to center. These images show what a recovery looks like starting from left to center :

![alt text][image3]

right to center:

![alt text][image4]

To augment the data set, I also flipped images and used left and right camera images for training with little steering angle correction factor 0.2.

Below is the original and it's corresponding flipped image:

![alt text][image5]

After the collection process, I had 11952 number of data points. I then preprocessed this data by and because i used left and right camera images so total data images 11952 X 3 = 35856. And then flipped all the images. so total data images 35856 X 2 = 71712

I finally randomly shuffled the data set and put 20% of the data into a validation set. 
So training data = 57369 withh validation data = 14343

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 10. Which also depends on number of data. I used an adam optimizer so that manually training the learning rate wasn't necessary.

Here is the video recording of vehicle driving autonomously 1 lap around the track.
First i used the code
```
python drive.py model.m5 run73

```
run73 stores all the images seen by agent with timestamp.Then i used below code to create video **run73.mp4** from generated images.

```
python video.py run73

```

<a href='run73.mp4'>run.mp4</a>
<video width="420" height="210" controls src='run73.mp4' />

