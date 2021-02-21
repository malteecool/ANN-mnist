import numpy as np
import matplotlib.pyplot as plt
"""
The image is transformed to a np-array of 28x28 "pixels".
"""


"""
Class to hold data for image and repective label.
The image is given in a linear array of ascii values.
"""
class image:

  def __init__(self, image, label):
    self.image = np.array(image)
    self.label = label


  def transform(self):
        return self.image.reshape(28, 28)

# Print grayscaled image in graph.
# Image to be a linear array of ascii data.
  def print_image(self):
      plt.imshow(transform(), cmap="Greys")
      print(self.label)
      plt.show()
