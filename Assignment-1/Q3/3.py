#Imports
import numpy as np
import cv2
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.image as img
from math import sqrt, log
from scipy.ndimage import gaussian_laplace, gaussian_filter
from skimage.feature import peak_local_max, blob_dog
from skimage.feature.blob import _prune_blobs

#Functions
def laplacian_of_gaussian(image, min_sigma=1, max_sigma=50, num_sigma=10, k=2):
    
    image = image.astype(np.float64)

    sigma_values = np.linspace(min_sigma, max_sigma, num_sigma)

    gaussian_laplace_images = [-gaussian_laplace(image, s) * s ** 2 for s in sigma_values]
    image_cube = np.dstack(gaussian_laplace_images)

    local_maxima = peak_local_max(image_cube, threshold_abs=.2, footprint=np.ones((3, 3, 3)), threshold_rel=0.0, exclude_border=False)

    if local_maxima.size == 0:
        return np.empty((0,3))

    lm = local_maxima.astype(np.float64)
    lm[:, 2] = sigma_values[local_maxima[:, 2]]

    return _prune_blobs(lm, .5)

def difference_of_gaussian(image, min_sigma=1, max_sigma=50, sigma_ratio=1.6):

    image = image.astype(np.float64)

    # k such that min_sigma*(sigma_ratio**k) > max_sigma
    k = int(log(float(max_sigma) / min_sigma, sigma_ratio)) + 1

    # a geometric progression of standard deviations for gaussian kernels
    sigma_list = np.array([min_sigma * (sigma_ratio ** i) for i in range(k + 1)])

    gaussian_images = [gaussian_filter(image, s) for s in sigma_list]

    dog_images = [(gaussian_images[i] - gaussian_images[i + 1]) for i in range(k)]
    image_cube = np.dstack(dog_images)

    local_maxima = peak_local_max(image_cube, threshold_abs=0.1, footprint=np.ones((3, 3, 3)), threshold_rel=0.0, exclude_border=False)
   
    if local_maxima.size == 0:
        return np.empty((0,3))
   
    lm = local_maxima.astype(np.float64)
    lm[:, 2] = sigma_list[local_maxima[:, 2]]
  
    return _prune_blobs(lm, .5)

#Code
image = cv2.imread('/Users/yashsrivastava/Documents/Files/CV-Assignments/Assignment-1/Images/Input/HW1_Q3/butterfly.jpg',0)

blobs_log = laplacian_of_gaussian(image)
blobs_log[:, 2] = blobs_log[:, 2] * sqrt(2)

blobs_dog = difference_of_gaussian(image, max_sigma=30)
blobs_dog[:, 2] = blobs_dog[:, 2] * sqrt(2)

blobs_list = [np.empty((0,3)),blobs_log, blobs_dog]
colors = ['blue','red', 'yellow']
titles = ['Original Image','Laplacian of Gaussian', 'Difference of Gaussian']
sequence = zip(blobs_list, colors, titles)

fig, axes = plt.subplots(1, 3, figsize=(9, 3), sharex=True, sharey=True, subplot_kw={'adjustable': 'box-forced'})
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
