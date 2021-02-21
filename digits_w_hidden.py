import numpy as np
import pandas as pd
from image_data import image
from scipy.stats import truncnorm
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import sys
import random


"""
Sigmond activation function.
"""
def sigmoid(z):
    return 1.0 / (1 + np.exp(-z))

"""
Truncated function: https://www.python-course.eu/neural_network_weights.php

Create random number with normal distribution.
If no other value is given to the function, the mean
is set to 0. 
The range of the values generated are dependant 
on the number of nodes in the layer of 
the network. (usually -0.1 to 0.1)
"""
def truncated(mean=0, sd=1, low=0, upp=10):
    return truncnorm((low - mean) / sd,
                     (upp - mean) / sd,
                     loc=mean,
                     scale=sd)


"""
Convert data to a "one_hot" representation.
Each integer value is represented as a array of
binary values, every value is 0 exept the index
of the integer. 
Example: 
8 = [0 0 1 0]
9 = [0 1 0 0]
"""
def data_as_one_hot(label_array, data):
    one_hot = (label_array == data).astype(np.float)
    one_hot[one_hot == 0] = 0.01
    one_hot[one_hot == 1] = 0.99
    return one_hot

"""
Function to load the data from the given image and 
label files.

Note: The files needs to be in ascii-format. 

The function reads every lines in the image 
and label file and store the image-data
with the corresponding label in a own 
data-structure.

The image-data is stored with values ranging
between 0 and 255, to facitilate the dot-product
with the layers weights during training, the values of
the image data is converted to range between 0.01 and 
0.99.

The data is divided the test size passed to the function.
Example: If 0.25 is passed as test size, 25% of the 
data will be used to test, and 75 will be used for
training. The data is also shuffled in a random order,
this is done to let the perceptron train of different
data each time. But when analyzing the functionality
and accuracy of the perceptron and the program as a
whole, a seed is used. This will cause the data
to be randomized "in the same order" each time.
"""
def load_data(image_file, label_file, test_size):
        
    training_images = open(image_file)
    training_label = open(label_file)

    for i in range(2):
        training_images.readline()
        training_label.readline()


    img_train_size, cols, rows, numb_images = training_images.readline().split()
    lbl_train_size, numb_labels = training_label.readline().split()
    
    image_size = int(cols) * int(rows)

    if(numb_images != numb_labels and img_train_size != lbl_train_size):
        print("#Invalid files")
        exit()

    data = []
    for i in range(int(img_train_size)):
        tbuffer = training_images.readline().split(" ")
        try:
            temp = int(training_label.readline()[0])
        except:
            # print("#error")
            exit()

        try:
            tbuffer.remove('\n')
            vbuffer.remove('\n')
        except:
            pass
        if(tbuffer[0] != ''):
            img = image(list(map(int, tbuffer)), temp)
            data.append(img)

    random.Random(628875008).shuffle(data)

    X = [data[i].image for i in range(len(data))]
    Y = [data[i].label for i in range(len(data))]

    fac = 0.99 / 255
    X = np.asfarray(X) * fac + 0.01

    data_size = len(data) - (len(data)*test_size)
    
    x_train, x_test = X[:int(data_size)], X[int(data_size):]
    y_train, y_test = Y[:int(data_size)], Y[int(data_size):]
    
    y_train = np.array(y_train,ndmin=2).T
    y_test = np.array(y_test, ndmin=2).T

    return x_train, x_test, y_train, y_test, image_size
    #return train_test_split(X, Y, test_size=test_size, random_state=628875008)

"""
Class to represent a artificial neural network. 
The network is used to recognize handwritten images
from a ascii representation of values ranging from
0 to 255. 

The network trains in multiple epochs until a acceptable
accuracy is achieved.

The data is loaded from the included training-image/label files. 
"""
class ANN:

    """
    Initializes the network with the number of nodes on each layer.
    Learninig rate is used for updating the weights of the network.
    """
    def __init__(self,
                 input_nodes,
                 output_nodes,
                 hidden_nodes,
                 learning_rate):
        self.input_nodes = input_nodes
        self.output_nodes = output_nodes
        self.hidden_nodes = hidden_nodes
        self.learning_rate = learning_rate


    """
    Function to load the validation images to test on the
    network.

    Image_file: file of images in ascii format. 

    The function read all lines in the file and maps it to a
    array. The values read from the file range between
    0 and 255, these values are then converted to range 
    between 0.01 and 0.99. This favours when using the 
    dot-product for the images and the weights. 

    The first three lines contain comments and other information
    of how the file is structured and are skipped. 
    """
    def load_validation(self, image_file):
        val_image = open(image_file)
        
        for i in range(2):
            val_image.readline()

        n_images, cols, rows, numbers = val_image.readline().split()

        data = []

        for i in range(int(n_images)):
            buffer = val_image.readline().split(" ")
            try:               
                buffer.remove('\n')
            except:
                pass
            if(buffer[0] != ''):
                temp = (list(map(int, buffer)))
                fac = 0.99 / 255
                data.append([(i * fac + 0.01) for i in temp])

        return data

    """
    Sets the weights of the neural network.
    Rows = number of neurons in the previous layer
    Cols = number of neurons in the next layer. 

    Each layer node has a certain weight to proceed to the next layer.
    Truncnorm used to get the normal distribution of values.
    The weights are initialized with random variables from the
    given size from truncnorm.
    """
    def matrice(self):

        rad = 1 / np.sqrt(self.input_nodes)
        X = truncated(mean=0, sd=1, low=-rad, upp=rad)
        self.win = X.rvs((self.hidden_nodes, self.input_nodes))

        rad = 1 / np.sqrt(self.hidden_nodes)
        X = truncated(mean=0, sd=1, low=-rad, upp=rad)
        self.wout = X.rvs((self.output_nodes, self.hidden_nodes))
        

    """
    Train the network with the training images and labels.
    The given paramters will be transformed into a two-
    dimensional array (vector). 

    x_train: array of image data.
    y_train: array of labels.

    We calculate the the first output vector by multiplying 
    the image vector by the weights (dot-product).
    The sigmoid activation function is used to calculate 
    the neurons output. 

    We do the same for the second output vector and calculate
    the error by subtracting the actual label from the expected
    label. 

    To update the weights, first the sigmoid derivative of the
    networks output is calculated, this is also multiplied by the
    outputs error. Finally the first layer vector and the newly 
    calculated vector are multiplied using the dot-product. 
    This is added to the weights.

    The procedure is repeted for the first layer weights.

    The basic approach is, if the network produces a large error,
    the prediction is bad and the network are updated with larger
    weights. If, on the otherhand, the network produces a close
    predicton, which results in a "smaller" error and better
    weights. 

    """
    def train(self, x_train, y_train):
        x_train = np.array(x_train, ndmin=2).T
        
        y_train = np.array(y_train, ndmin=2).T

        try: 
            # Input -> hidden
            out1 = np.dot(self.win, x_train)
            hidden = sigmoid(out1)

            # Hidden -> output
            out2 = np.dot(self.wout, hidden)
            network = sigmoid(out2)
        except:
            print("# Could not sum the arrays, maybe invalid size?")
            exit()

        errors = y_train - network

        # Sigmoid derivative
        tmp = network \
            * (1.0 - network)

        tmp = errors * tmp
        tmp = self.learning_rate * np.dot(tmp, hidden.T)
        self.wout += tmp

        hidden_errors = np.dot(self.wout.T,
                               errors)

        tmp = hidden * \
            (1.0 - hidden)

        tmp = hidden_errors * tmp
        self.win += self.learning_rate \
            * np.dot(tmp, x_train.T)

    """
    Function used to present the accuracy. 

    The function takes all images and labels, and 
    "runs" them. Here all results of the prediction
    is saved. If the label is the same as the networks
    prediction, the amound of correct guesses will be 
    increased, else the error will be increased. 

    The result returned from the run-function
    gives an array of predictions from all
    output neurons. The one with the highest
    value is picked as the best prediction.
    """
    def evaluate(self, test_images, test_labels):
        correct = 0
        error = 0
        for i in range(len(test_images)):
            res = self.run(test_images[i])
            predict = labels[res.argmax()]
            
            if predict == test_labels[i]:
                correct += 1
            else:
              error += 1
            
        return correct, error

    """
    Function to "run" and predict a image.

    The given test image is transformed into a
    vector. 

    The same procedure as in the training is done, 
    without updating the weights. 

    The result is an array of values given the networks 
    "output layer". This layer consists of 4 neurons
    which each gives a predicion of the image.

    """ 
    def run(self, test_images):

        test_images = np.array(test_images, ndmin=2).T
        
        #input -> hidden
        res = np.dot(self.win, test_images)

        res = sigmoid(res)

        #hidden -> output
        res = np.dot(self.wout, res)

        res = sigmoid(res)

        return res



labels = [4, 7, 8, 9]
#image_size = 28*28
out_put_nodes = len(labels)


"""
The test size is set to 0.25, which means 75% of the data will be used
for training. 
The files should be passed in as arguments to the program.
"""
x_train, x_test, y_train, y_test, image_size = load_data(sys.argv[1], sys.argv[2], 0.75)

"""
The network is set to have a input layer of 784 neurons, the
pixels of the image. Since there is only 4 labels present in the
testing-files, the output will only consist of these 4 neurons.
Training size of 0.1.
"""
ann = ANN(image_size, out_put_nodes, 100, 0.1)


# Init matrice
ann.matrice()

"""
Convert the labels to a one-hot representation. 
"""
one_hot = data_as_one_hot(labels, y_train)

"""
The network runs in multiple epochs, until the
treshhold for the accuracy is achieved. 

After each training-round, the accuracy is evaluated
by running on the test-set.
"""
epoch = 1
accuracy = 0
accuracy_scores = []
while accuracy < 0.9:
  for i in range(len(x_train)):
      ann.train(x_train[i], one_hot[i])

  correct, error = ann.evaluate(x_test, y_test)
  accuracy = correct/(correct+error)
  accuracy_scores.append(accuracy)
  #print("# Accuracy of epoch ", epoch, ": ", accuracy)
  epoch += 1

#print("# Final accuracy: ", accuracy)


# Create x-axis
#xaxis = [x for x in range(epoch-1)]

# Plot the test and training scores with labels
#plt.plot(xaxis, accuracy_scores, label='Accuracy of epoch')
#plt.plot(xaxis, test_scores, label='Test score')

# Show the figure
#plt.legend()
#plt.show()

"""
***********************************************************
When the acceptable accuracy is achieved, 
the validation-images are loaded. The prediction of these
are printed to stdout. This is only done for 
the submission on labres and not necessary for the network.
***********************************************************
"""
val_images = ann.load_validation(sys.argv[3])
for i in range(len(val_images)):
    res = ann.run(val_images[i])
    print(labels[np.argmax(res)])
