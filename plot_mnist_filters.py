"""
=====================================
Visualization of MLP weights on MNIST
=====================================

Sometimes looking at the learned coefficients of a neural network can provide
insight into the learning behavior. For example if weights look unstructured,
maybe some were not used at all, or if very large coefficients exist, maybe
regularization was too low or the learning rate too high.

This example shows how to plot some of the first layer weights in a
MLPClassifier trained on the MNIST dataset.

The input data consists of 28x28 pixel handwritten digits, leading to 784
features in the dataset. Therefore the first layer weight matrix have the shape
(784, hidden_layer_sizes[0]).  We can therefore visualize a single column of
the weight matrix as a 28x28 pixel image.

To make the example run faster, we use very few hidden units, and train only
for a very short time. Training longer would result in weights with a much
smoother spatial appearance. The example will throw a warning because it
doesn't converge, in this case this is what we want because of CI's time
constraints.
"""

import warnings

import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.exceptions import ConvergenceWarning
from sklearn.neural_network import MLPClassifier


def load_data():

    training_images = open("training-images.txt")
    training_label = open("training-labels.txt")

    validation_images = open("validation-images.txt")

    validation_labels = open("validation-labels.txt")

    #Read the first to commented lines
    for i in range(2):
        training_images.readline()
        training_label.readline()
        validation_images.readline()
        validation_labels.readline()

    img_train_size, cols, rows, numb_images = training_images.readline().split()
    lbl_train_size, numb_labels = training_label.readline().split()

    img_val_size, cols_train, rows_train, numb_images_train = validation_images.readline().split()
    lbl_val_size, numb_labels_train = validation_labels.readline().split()

    # Check if the line number is matching.
    if(numb_images != numb_labels and img_train_size != lbl_train_size):
        print("#Invalid files")
        exit()

    train_images = []
    train_labels = []

    test_images = []
    test_labels = []
    print("#", numb_images)
    for i in range(int(img_train_size)):
        tbuffer = training_images.readline().split(" ")
        vbuffer = validation_images.readline().split(" ")
        try:
            train_labels.append(int(training_label.readline()[0]))
            test_labels.append(int(validation_labels.readline()[0]))
        except:
            print("#error!!")
            exit()

        try:
            tbuffer.remove('\n')
            vbuffer.remove('\n')
        except:
            pass
        if(tbuffer[0] != '' and vbuffer[0] != ''):
            train_images.append(list(map(int, tbuffer)))
            test_images.append(list(map(int, vbuffer)))

    return (np.array(train_images), np.array(train_labels)), (np.array(test_images), np.array(test_labels))

# rescale the data, use the traditional train/test split
X_train, X_test, y_train, y_test = load_data()

X_train = X_train / 255.0
y_train = y_train / 255.0

mlp = MLPClassifier(hidden_layer_sizes=(50,), max_iter=10, alpha=1e-4,
                    solver='sgd', verbose=10, random_state=1,
                    learning_rate_init=.1)

# this example won't converge because of CI's time constraints, so we catch the
# warning and are ignore it here
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=ConvergenceWarning,
                            module="sklearn")
    mlp.fit(X_train, y_train)

print("Training set score: %f" % mlp.score(X_train, y_train))
print("Test set score: %f" % mlp.score(X_test, y_test))

fig, axes = plt.subplots(4, 4)
# use global min / max to ensure all weights are shown on the same scale
vmin, vmax = mlp.coefs_[0].min(), mlp.coefs_[0].max()
for coef, ax in zip(mlp.coefs_[0].T, axes.ravel()):
    ax.matshow(coef.reshape(28, 28), cmap=plt.cm.gray, vmin=.5 * vmin,
               vmax=.5 * vmax)
    ax.set_xticks(())
    ax.set_yticks(())

plt.show()
