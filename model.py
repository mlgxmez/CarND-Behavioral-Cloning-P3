import csv
from scipy.misc import imread
import numpy as np

# Function to load an image from its path
def arrange_img(img_path):
	img_name = img_path.split('/')[-1]
	path_rel_img = './data/IMG/' + img_name
	read_img = imread(path_rel_img)
	return read_img

# Generator to serve data to the model in batches
def generator(samples_X, samples_y, batch_size=32):
	num_samples = len(samples_X)
	while 1:
		seq_samples = np.random.permutation(num_samples)
		for offset in range(0, num_samples, batch_size):
			idx_samples = seq_samples[offset:offset+batch_size]
			batch_X = samples_X[idx_samples,:,:,:]
			batch_y = [samples_y[i] for i in idx_samples]
		yield (batch_X, batch_y)


lines = []

# Append path to images to a list
with open('./data/driving_log.csv') as csvfile:
	reader = csv.reader(csvfile)
	next(reader) # Delete this line if you are not using the Udacity data
	for line in reader:
		lines.append(line)

images = []
steering_angles = []

for ll in lines:
	# The first three columns contain center, left and right image paths
	image_frame = [arrange_img(ll[i]) for i in range(3)]
	images = images + image_frame
	steering = float(ll[3])
	steering_angles = steering_angles + [steering, steering + 0.2, steering - 0.2]

X_data = np.array(images)
y_data = steering_angles


# Split data into train and validation sets
from sklearn.model_selection import train_test_split
X_train, X_validation, y_train, y_validation = train_test_split(X_data, y_data, test_size=0.2)
batch_size = 128
train_generator = generator(X_train, y_train, batch_size)
validation_generator = generator(X_validation, y_validation, batch_size)

from keras import optimizers
from keras.models import Model
from keras.layers import Dense, MaxPooling2D, Input, Cropping2D, Dropout, Lambda, Flatten
from keras.layers.convolutional import Convolution2D
from keras.utils.visualize_util import plot

# Input and preprocessing layers
img_tensor_0 = Input(shape=(160, 320, 3))
img_tensor_1 = Lambda(lambda x: (x/255.0) - 0.5)(img_tensor_0)
img_tensor_2 = Cropping2D(cropping=((50,20), (0,0)))(img_tensor_1)

# Neural network architecture
x = Convolution2D(6,5,5, activation='relu',border_mode='valid')(img_tensor_2)
x = MaxPooling2D()(x)
x = Convolution2D(6,5,5, activation='relu',border_mode='valid')(x)
x = MaxPooling2D()(x)
x = Convolution2D(12,5,5, activation='relu',border_mode='valid')(x)
x = MaxPooling2D()(x)
x = Convolution2D(32,3,3, activation='relu',border_mode='valid')(x)
x = MaxPooling2D()(x)
x = Flatten()(x)
x = Dense(512, activation='relu')(x)
x = Dropout(0.5)(x)
x = Dense(256, activation='relu')(x)
x = Dropout(0.5)(x)
x = Dense(32, activation='relu')(x)

# Output layer
predictions = Dense(1)(x)

# Build the model
model = Model(input=img_tensor_0, output=predictions)

plot(model, to_file='model.png', show_shapes = True)

# Define the loss function and the optimizer
model.compile(loss='mse', optimizer='rmsprop')

# Train the model using the generator function
model.fit_generator(train_generator, samples_per_epoch=(len(X_train)//batch_size)*batch_size, validation_data=validation_generator, nb_val_samples=len(X_validation), nb_epoch=4)

# Save the model to drive the car in autonomous mode
model.save('model.h5')
print("Keras model has been saved")