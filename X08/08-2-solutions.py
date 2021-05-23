# You can find such "simple" examples on the internet a dime a dozend. 
# This problem set is based off the code/explanations you can find here:
# https://nextjournal.com/gkoehler/digit-recognition-with-keras

# ============================================================================ #
# Dependencies

# Our "standard libraries"
import numpy as np
import matplotlib.pyplot as plt
import os

# Deep Learning Classes and functions
from keras.datasets import mnist
from keras.models   import Sequential
from keras.layers   import Dense, Activation, Dropout
from keras.utils    import np_utils

# ============================================================================ #
# Behaviour Constants

save_dir = "./results/"
model_name = 'keras_mnist.h5'
model_path = os.path.join(save_dir, model_name)

if not os.path.exists(save_dir) : os.mkdir(save_dir )

# ============================================================================ #
# Load MINST handwritten digits dataset

(X_train, y_train), (X_test, y_test) = mnist.load_data()                        # running this the first time will download the numbers from the internet

# ============================================================================ #
# Data Preprocessing

# get how many pictures are in each set...
N_train = X_train.shape[0]
N_test  = X_test .shape[0]

# ... and what their resolution is. We need only one variable here, since
# our project by design can only work with pictures of same resolution.
resolution = X_train[0].shape

# later we'll need the number of pixels per picture:
N_pixels = np.prod(resolution)

# normalizing the data to help with the training
# Learning works best (fastest) if the magnitude of activations of all neurons
# is about the same.
# Our input is Pixel brightness and is in the range 0..255 ("1 Byte worth of
# options"), while our output is in the range 0..100 (percent probability of the
# picture representing some digit). Hence, dividing the input brightness by its
# maximum value is beneficial to the learning process
# However, the following does not work directly:

#X_train /= 255
#X_test  /= 255

# this is because X_... are numpy INTEGER arrays. Dividing them would make 
# FLOATs out of them, and numpy forbids implicit type conversions. Instead, we
# first have to make the conversion ourselves before normalization works:

X_train = X_train.astype(np.float32)
X_test  = X_test .astype(np.float32)

X_train /= 255
X_test  /= 255

# Further, our NN expects an input VECTOR, so we have to flatten our pictures.
# building the input vector from the 28x28 pixels
X_train = X_train.reshape(N_train, N_pixels)
X_test  = X_test .reshape(N_test , N_pixels)


# Our Output y is still in the wrong format: we want a vector where y[i] is the
# probability of picture y showing the digit i. We could do this by hand:

#y_new = np.zeros( (N_test, 10) )
#for i in range(N_test) :
#  y_new[i, y_test[i]] = 1
#
#print(y_test[0])
#print(y_new[0])

# or we use the built-in tool np_utils.to_categorical from keras:
N_classes = 10
Y_train = np_utils.to_categorical(y_train, N_classes)
Y_test  = np_utils.to_categorical(y_test , N_classes)

# note how the lower case y is the "raw data" while the upper case Y stands for
# our preprocessed vectors.

# ============================================================================ #
# Defining the NN

model = Sequential()
model.add(Dense(16, input_shape=(N_pixels,)))
model.add(Activation('relu'))                            
model.add(Dropout(0.2))

model.add(Dense(16))
model.add(Activation('relu'))
model.add(Dropout(0.2))

model.add(Dense(10))
model.add(Activation('softmax'))

# compiling the sequential model
model.compile(
    loss='categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy'],
)


# ============================================================================ #
# training the model and saving metrics in history

print("#" * 80)
print("TRAINING BEGINS")

history = model.fit(
    X_train, Y_train,
    batch_size = 128,
    epochs     =  20,
    verbose    =   2,
    validation_data=(X_test, Y_test)
)

print("#" * 80)
print("TRAINING COMPLETED")

model.save(model_path)
print(f"Saved trained model at {model_path} ")

# ============================================================================ #
# Visualizing the training success

fig = plt.figure()

plt.subplot(2,1,1)
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.plot(history.history['accuracy']    , label='training')
plt.plot(history.history['val_accuracy'], label='test')
plt.legend()

plt.subplot(2,1,2)
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.plot(history.history['loss']    , label='training')
plt.plot(history.history['val_loss'], label='test')
plt.legend()

plt.tight_layout()
plt.show()

# ============================================================================ #
# Debug Helper

# keras is somewhat verbose and will clutter the console with all sorts of 
# messages. It is therefore convenient to make sure we see in a singe glance
# whether our last few lines actually worked or triggered some error code.
# To that end we can either make sure our last output lines always look the
# same:

print("#" * 80)
print("ALL DONE.")

# or we can "silence" keras:

#os.environ['TF_CPP_MIN_LOG_LEVEL']='3'

# The messages of keras all have an "importance" assigned. With the above line
# we can tell keras only to tell us everything that is at least of level 3
# importance.
# Of course, for this to work, we need to place the line at the very beginning
# of our script, right after we've loaded os.
