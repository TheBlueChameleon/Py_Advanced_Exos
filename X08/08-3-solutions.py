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
from keras.models   import load_model
from keras.utils    import np_utils

# ============================================================================ #
# Behaviour Constants

save_dir = "./results/"
model_name = 'keras_mnist.h5'
model_path = os.path.join(save_dir, model_name)

# ============================================================================ #
# Load MINST handwritten digits dataset and repeat preprocessing

# essentially, loading the *_test objects would suffice, but in this case,
# being a bit wasteful and loading everything is just simpler ;)

(X_train, y_train), (X_test, y_test) = mnist.load_data()

# We will at least limit the preprocessing to the *_test data.
resolution = X_test[0].shape
N_test     = X_test.shape[0]
N_pixels   = np.prod(resolution)

X_test     = X_test.reshape(N_test , N_pixels)
X_test     = X_test.astype(np.float32)
X_test    /= 255

N_classes  = 10
Y_test     = np_utils.to_categorical(y_test , N_classes)

# ============================================================================ #
# Load the previously loaded model

print("#" * 80)
print("ABOUT TO LOAD MODEL...")

model = load_model(model_path)

print("... DONE!")

# ============================================================================ #
# Evaluate the accuracy of the loaded model:

print("#" * 80)
print("EVALUATION:")
print()

print("automated:")
loss_and_metrics = model.evaluate(X_test, Y_test, verbose=2)

print()
print("Test Loss    :", loss_and_metrics[0])
print("Test Accuracy:", loss_and_metrics[1])
print()

print("manual:")
predicted_classes = np.argmax(
    model.predict(X_test),
    axis=1
).astype(np.int32)
    # read this from the inside out:
    # model.predict tries to guess the digit shown in the images X and returns a
    # vector y with the probabilites for the respective numbers.
    # argmax now finds the digit with the highest probability per line (axis=1)
    # and hence returns a list of indices i.e. a list of guessed digits.
    
correct_indices   = np.nonzero(predicted_classes == y_test)[0]
incorrect_indices = np.nonzero(predicted_classes != y_test)[0]
    # np.nonzero(a) returns a tuple of np.ndarrays.
    # the tuple-index is the dimension in a at which the ndarray-index should be
    # inserted.
    # so len(np.nonzero(a)) == len(a.shape)
    # here we have a 1D list of booleans, so np.nonzero(...)[0] tells us exactly
    # for which images the comparison evaluated to true.
print("(see plot window)")

# ============================================================================ #
# Evaluate the accuracy of the loaded model:

figure_evaluation = plt.figure( figsize=(7,14) )

# plot 9 correct predictions
for i, correct in enumerate(correct_indices[:9]):
    plt.subplot(6,3,i+1)
    plt.imshow(
        X_test[correct].reshape(28,28),
        cmap='gray',
        interpolation='none'
    )
    plt.title(f"Predicted: {predicted_classes[correct]}, Truth: {y_test[correct]}", color="green")
    plt.xticks([])
    plt.yticks([])

# plot 9 incorrect predictions
for i, incorrect in enumerate(incorrect_indices[:9]):
    plt.subplot(6,3,i+10)
    plt.imshow(
        X_test[incorrect].reshape(28,28),
        cmap='gray',
        interpolation='none'
    )
    plt.title(f"Predicted {predicted_classes[incorrect]}, Truth: {y_test[incorrect]}", color="red")
    plt.xticks([])
    plt.yticks([])

plt.show()

# ============================================================================ #
# Debug Helper

print("#" * 80)
print("ALL DONE.")
