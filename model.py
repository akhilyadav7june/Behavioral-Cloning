import csv
import cv2
import numpy as np

lines = []

#Adding all the rows from csv file in lines list
with open("./data/driving_log.csv") as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)

#Because lines first item contains the heading so removing that
lines.remove(lines[0])

#images list is used to store all images and measurements list is used to store steering angle
images = []
measurements = []

j=0

#This loop is used to store images captured from center , left and right camera in images list
# Also storing the steering angles in measurements list 
for line in lines:
	for i in range(3):
		source_path = line[i]
		filename = source_path.split("/")[-1]
		current_path = "./data/IMG/" + filename
		if(j==0):
			print(current_path+''+line[3])
			j=j+1
		image = cv2.imread(current_path)
		images.append(image)

	steering_center = float(line[3])

	#Adding the 0.2 correction factor for left and right camera
	correction = 0.2
	steering_left = steering_center + correction
	steering_right = steering_center - correction

        #Adding steering angle in measurement list
	measurements.append(steering_center)
	measurements.append(steering_left)
	measurements.append(steering_right)

# Augmenting the data by flipping all images and also for steering angle multiplying by -1.
augmented_images, augmented_measurements = [], []
for image,measurement in zip(images, measurements):
   augmented_images.append(image)
   augmented_measurements.append(measurement)
   augmented_images.append(cv2.flip(image,1))
   augmented_measurements.append(-1.0*measurement)

#Converting the data in numpy array to be used by keras model
X_train = np.array(augmented_images)
y_train = np.array(augmented_measurements)

#importing all the required libraries
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D, Dropout
from keras.layers.convolutional import Conv2D
from keras.optimizers import Adam

#Creating the model
model = Sequential()

#Adding lambda layer for normalizing the data
model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(160,320,3)))

#Cropping the uneccessary image data from upper and bottom of the image
model.add(Cropping2D(cropping=((70,25),(0,0))))

#Convolution layers, Fully connected and dropout layer
model.add(Conv2D(24,(5,5),strides=(2,2),activation="relu"))
model.add(Conv2D(36,(5,5),strides=(2,2),activation="relu"))
model.add(Conv2D(48,(5,5),strides=(2,2),activation="relu"))
model.add(Conv2D(64,(3,3),activation="relu"))
model.add(Conv2D(64,(3,3),activation="relu"))
model.add(Flatten())
model.add(Dense(100))
model.add(Dropout(0.5))
model.add(Dense(50))
model.add(Dropout(0.5))
model.add(Dense(10))
model.add(Dropout(0.5))

#Output layer for predicting steering angle 
model.add(Dense(1))

#Using mean squred error with Adam optimizer
model.compile(loss='mse', optimizer='adam')

#Printing the model summary during execution
model.summary()

#Training the model with 80% data with epochs 10 and shuffling 
model.fit(X_train, y_train, validation_split=0.2, shuffle=True, epochs=10)

#Saving the trained model
model.save('model.h5')
