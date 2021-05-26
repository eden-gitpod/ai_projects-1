#!/usr/bin/env python
# coding: utf-8


# Import the necessary libs
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import RMSprop
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D, BatchNormalization
from keras.models import Sequential
import tensorflow as tf
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt

# ! to generate the same random numbers everytime we run the script
np.random.seed(2)


#! ------------------------- Shuffle & Split the dataset -----------------------

#! keras has the dataset already preprocessed so we can use it directly
(imgs_train, labels_train), (imgs_test,
                             labels_test) = tf.keras.datasets.mnist.load_data()

# Preprocess the data (these are NumPy arrays)
imgs_train = imgs_train.reshape(-1, 28, 28, 1).astype("float32") / 255
imgs_test = imgs_test.reshape(-1, 28, 28, 1).astype("float32") / 255

labels_train = labels_train.astype("float32")
labels_test = labels_test.astype("float32")

#! splitt and shuffle the data sothat the test data is 30% of the whole data
x_train, x_val, y_train, y_val = train_test_split(
    imgs_train, labels_train, test_size=0.3)


def show_img(x_data, y_data, index, show_axis='off', cmap="gray"):
    plt.axis(show_axis)
    plt.suptitle(y_data[index], fontsize=20)
    plt.imshow(x_data[index][..., 0],  cmap=cmap)


show_img(x_train, y_train, 14301)


#! ------------------------- Augmenting the Images -----------------------------
# we will apply this get get more images for traning by applying some image processing techniques
# like zooming, shifting, etc...

datagen = ImageDataGenerator(
    zoom_range=0.1,  # Randomly zoom image
    # randomly shift images horizontally (fraction of total width)
    width_shift_range=0.1,
    # randomly shift images vertically (fraction of total height)
    height_shift_range=0.1,
    horizontal_flip=False,  # don't flip images
    vertical_flip=False,  # don't flip images
)


datagen.fit(x_train)


#! ------------------------------- Create the model ----------------------------

# Design the model

model = Sequential([
    # -------------- Convolution Layers --------------

    # First CONV
    Conv2D(64, (3, 3), activation="relu",
           input_shape=(28, 28, 1), padding="same",),
    MaxPool2D((2, 2)),
    BatchNormalization(),

    # Second CONV
    Conv2D(64, (3, 3), activation="relu", padding="same",),
    BatchNormalization(),
    MaxPool2D((2, 2)),
    Dropout(0.3),

    # Third CONV
    Conv2D(128, (3, 3), activation="relu", padding="same",),
    BatchNormalization(),
    MaxPool2D((2, 2)),
    MaxPool2D((2, 2), strides=(2, 2)),
    Dropout(0.3),


    # -------------- Fully Connected Layers --------------
    Flatten(),
    Dense(256, activation="relu"),
    BatchNormalization(),
    Dropout(0.3),


    # Output Layer we will predict 10 numbers from 0 to 9
    Dense(10, activation="softmax"),

])


# Compile the model
optimizer = RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)
loss = "sparse_categorical_crossentropy"
model.compile(optimizer=optimizer, loss=loss, metrics=["acc"])


#! ------------------------- Start Model Training ------------------------------

class myCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        # keep track of the validation accuracy and if it reaches high accuracy like 99%
        if(logs.get('val_acc') >= 0.99):
            print("\nReached 99% accuracy so cancelling training!")
            self.model.stop_training = True


callbacks = myCallback()


epochs = 32  # Turn epochs to 30 to get 0.9967 accuracy
batch_size = 256  # 64,128 ,256,512

with tf.device("/GPU:0"):

    # Prediction model
    history = model.fit_generator(
        datagen.flow(x_train, y_train, batch_size=batch_size),
        epochs=epochs,
        validation_data=(x_val, y_val),
        verbose=1,
        steps_per_epoch=x_train.shape[0] // batch_size,
        callbacks=[callbacks]
    )


#! --------------------------- Save the model ----------------------------------

model.save('english_digits_model')  # creates a pb file (Saved model)
get_ipython().system('zip -r eng.zip "./english_digits_model"')


#! ---------------------------------- Test -------------------------------------
show_img(x_train, y_train, 1555)


print(model.predict(x_train[1555].reshape(1, 28, 28, 1)).argmax())
