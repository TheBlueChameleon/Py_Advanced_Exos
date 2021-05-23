# You can find such "simple" examples on the internet a dime a dozend. 
# This problem set is based off the code/explanations you can find here:
# https://nextjournal.com/gkoehler/digit-recognition-with-keras

# ============================================================================ #
# Dependencies

# Deep Learning Classes and functions
from keras.datasets import mnist

# Our "standard libraries"
import matplotlib.pyplot as plt

# ============================================================================ #
# Load MINST handwritten digits dataset

# load the data to be analyzed, together with the proper labels
# running this the first time will download the numbers from the internet
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# In the following, X can be either X_train or X_test. Analogously, y means
# both, y_train and y_test.
# X are 3D arrays.
# The first index is the ID of the picture. That is, X[5] is the 6th picture 
# (indices start at 0) in the sample.
# The second and third index are the x and y coordinate of a pixel, respectively
# So, X[5, 4, 10] == 10 means that in picture 5 at coordinates (4, 10) the
# brightness value is 10.

# y is simply an array of labels associated with the pictures in X. If X[i] 
# shows the digit 5, then y[i] == 5.

# Show the first nine figures with their labels, just so we know what he have in
# our hands

fig = plt.figure()
for i in range(9):
    plt.subplot(3,3,i+1)
    
    plt.tight_layout()
    plt.imshow(X_train[i], cmap='gray', interpolation='none')
    plt.title("Digit: {}".format(y_train[i]))
    plt.xticks([])
    plt.yticks([])

plt.show()

# Use this codebase to play with the data ;)

# type(X), X.shape, type(X[i,j,k]), ... are things you should know when 
# beginning to work on the network itself.
