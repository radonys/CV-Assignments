#Imports
import numpy as np
import cv2
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.image as img
from math import sqrt
from skimage import data
from skimage.feature import blob_dog, blob_log

#Functions

#Code

image = cv2.imread('/Users/yashsrivastava/Documents/Files/CV-Assignments/Assignment-1/Images/Input/HW1_Q3/butterfly.jpg',0)

blobs_log = blob_log(image, max_sigma=30, num_sigma=10, threshold=.1)

# Compute radii in the 3rd column.
blobs_log[:, 2] = blobs_log[:, 2] * sqrt(2)

blobs_dog = blob_dog(image, max_sigma=30, threshold=.1)
blobs_dog[:, 2] = blobs_dog[:, 2] * sqrt(2)

blobs_list = [blobs_log, blobs_dog]
colors = ['red', 'yellow']
titles = ['Laplacian of Gaussian', 'Difference of Gaussian',]
sequence = zip(blobs_list, colors, titles)

fig, axes = plt.subplots(1, 2, figsize=(9, 3), sharex=True, sharey=True, subplot_kw={'adjustable': 'box-forced'})
ax = axes.ravel()

for idx, (blobs, color, title) in enumerate(sequence):
    ax[idx].set_title(title)
    ax[idx].imshow(image, interpolation='nearest', cmap = 'gray')
    for blob in blobs:
        y, x, r = blob
        c = plt.Circle((x, y), r, color=color, linewidth=2, fill=False)
        ax[idx].add_patch(c)
    ax[idx].set_axis_off()

plt.tight_layout()
plt.show()
