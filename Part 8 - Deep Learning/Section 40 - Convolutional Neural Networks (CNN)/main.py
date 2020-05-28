#Importing the libraries
import numpy as np
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
print(tf.__version__)

#Preprocessing the training set
#Image augmentation

trainDataGen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)

trainingSet = trainDataGen.flow_from_directory(
    'dataset/training_set',
    target_size=(64,64),
    batch_size=32,
    class_mode='binary'
)

#Preprocessing the test data, no image augmentation here as we want to predict for the actual photo

testDataGen = ImageDataGenerator(rescale=1./255)

testingSet = testDataGen.flow_from_directory(
    'dataset/test_set',
    target_size=(64,64),
    batch_size=32,
    class_mode='binary'
)

#Building the CNN

cnn = tf.keras.models.Sequential()

#Convolution
cnn.add(tf.keras.layers.Conv2D(filters=40, activation='relu', input_shape=[64,64,3], kernel_size=3))

#Pooling we'll use max pooling
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2,strides=2))

#Add second convolutional layer
cnn.add(tf.keras.layers.Conv2D(filters=40, activation='relu', kernel_size=3))
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2,strides=2))

#Flattening
cnn.add(tf.keras.layers.Flatten())

#Fully connected layer
cnn.add(tf.keras.layers.Dense(activation='relu',units=100))

#Output layer
cnn.add(tf.keras.layers.Dense(activation='sigmoid',units=1))

#Compiling the CNN
cnn.compile(optimizer= 'adam',metrics=['accuracy'],loss = 'binary_crossentropy')

#Training the CNN on the training set and evaluating it on the test set
cnn.fit(x=trainingSet,validation_data=testingSet,epochs=25)

#Making a single prediction
from keras.preprocessing import image
testImage = image.load_img('dataset/single_prediction/cat_or_dog_1.jpg',target_size=(64,64))
testImage = image.img_to_array(testImage)
testImage = np.expand_dims(testImage, axis=0)
result = cnn.predict(testImage)
if result[0][0] == 0:
    print("it is a cat")
else:
    print("it is a dog")