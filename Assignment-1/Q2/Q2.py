import cv2 as cv
import numpy as np
from numpy.linalg import eigvals
from numpy.lib.stride_tricks import as_strided
import matplotlib.pyplot as plt
import time
import argparse

"""
    Helper Functions
"""
GAUSSIAN_KERNEL=""
def gaussian_kernel(l=21, sig=1):
    """
    Creates gaussian kernel with side length l and a sigma of sig
    """
    ax = np.arange(-l // 2 + 1., l // 2 + 1.)
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(-(xx**2 + yy**2) / (2. * sig**2))
    return kernel / np.sum(kernel)

def windows_strided(image, window_size):
    """
    Generates stride matrices given image and window size
    """
    w, h = image.shape
    return as_strided(image,shape=[w - window_size + 1, h - window_size + 1, window_size, window_size],strides=image.strides + image.strides)

def eigenvals(matrix):
    """
    Calculates Eigen Values
    """
    gy, gx = np.gradient(matrix)
    M = np.zeros(shape=(2,2))
    for r in range(matrix.shape[0]):
        for c in range(matrix.shape[0]):
            M+=GAUSSIAN_KERNEL[r,c]*np.array([[gx[r,c]**2,gx[r,c]*gy[r,c]],[gy[r,c]*gx[r,c],gy[r,c]**2]])
    v = eigvals(M)
    return np.real(v)

def readimg(path):
    """
    Read image
    """
    y = cv.imread(path,0)
    y = y.astype('float')
    y/=np.amax(y)
    return y

"""
    Corner Detection Implementation
"""

def shiTomasi(img,window_size,threshold):
    """
    Score R:= min(eigenvalues)
    """
    start = time.process_time()
    print("Shi-Tomasi Corner Detection")
    coordinate_y = []
    coordinate_x = []
    strided = windows_strided(img,window_size)
    k = window_size//2
    l = window_size//2
    c = 0
    for stride in strided:
        for i in stride:
            if np.amin(eigenvals(i)) > threshold:
                c+=1
                coordinate_x.append(k)
                coordinate_y.append(l)
            k+=1
        k=window_size//2
        l+=1
    print("Corners Detected:{}".format(c))
    print("Total time: {}s".format(time.process_time()- start))
    return [coordinate_x,coordinate_y]

def harris(img,window_size,alpha,threshold):
    """
    Score R:= product(eigenvalues) - alpha*(sum(eigenvalues))
    """
    start = time.process_time()
    print("Harris Corner Detection")
    coordinate_y = []
    coordinate_x = []
    strided = windows_strided(img,window_size)
    k = window_size//2
    l = window_size//2
    c = 0
    for stride in strided:
        for i in stride:
            temp = eigenvals(i)
            R = np.subtract(np.prod(temp),np.multiply(alpha,np.sum(temp)))
            # if R>0:
            #     print(R)
            if R > threshold:
                c+=1
                coordinate_x.append(k)
                coordinate_y.append(l)
            k+=1
        l+=1
        k=window_size//2
    print("Corners Detected:{}".format(c))
    print("Total time: {}s".format(time.process_time()- start))
    return [coordinate_x,coordinate_y]
"""
    Main Program
"""
if __name__=="__main__":
    """
    Parser for Arguments
    usage: Q2.py [-h] [--image_path IMAGE_PATH] [--window_size WINDOW_SIZE]
             [--method METHOD] [--alpha ALPHA] [--threshold THRESHOLD]
    """
    parser = argparse.ArgumentParser(description="Corner detection algorithms")
    parser.add_argument('--image_path',help='Path to the image',type=str,default="chessboard.png")
    parser.add_argument('--window_size',help='window size of filter',type=int,default=7)
    parser.add_argument('--method',help='Type of algo to be used (shitomasi,harris,both)',type=str,default="both")
    parser.add_argument('--alpha',help='Alpha value for Harris',type=float,default=0.002)
    parser.add_argument('--thresholdH',help='threshold for cornerness score in Harris Corner Detector',type=float,default=0.0005)
    parser.add_argument('--thresholdS',help='threshold for cornerness score in Shi-Tomasi Corner Detector',type=float,default=0.01)
    args = parser.parse_args()

    img = readimg(args.image_path)
    original = cv.imread(args.image_path,1)
    GAUSSIAN_KERNEL = gaussian_kernel(args.window_size)
    coordinates_shi = []
    coordinates_har = []

    if args.method.lower()=='both' or args.method.lower()=='shitomasi':
        coordinates_shi = shiTomasi(img,args.window_size,args.thresholdS)

    if args.method.lower()=='both' or args.method.lower()=='harris':
        coordinates_har = harris(img,args.window_size,args.alpha,args.thresholdH)

    if args.method.lower()=='both':
        fig, (ax1, ax2, ax3) = plt.subplots(1,3)
        ax1.imshow(original)
        ax1.set_title("Input Image")
        ax2.imshow(original)
        ax2.plot(coordinates_shi[0],coordinates_shi[1],'.', color='firebrick')
        ax2.set_title("Shi-Tomasi Corner Detection")
        ax2.set_xlabel("Threshold: "+str(args.thresholdS))
        ax3.imshow(original)
        ax3.plot(coordinates_har[0],coordinates_har[1],'.', color='firebrick')
        ax3.set_title("Harris Corner Detection")
        ax3.set_xlabel("Threshold: "+str(args.thresholdH)+"\nAlpha: "+str(args.alpha))

    elif args.method.lower()=='harris':
        fig, (ax1, ax2) = plt.subplots(1,2)
        ax1.imshow(original)
        ax1.set_title("Input Image")
        ax2.imshow(original)
        ax2.plot(coordinates_har[0],coordinates_har[1],'.', color='firebrick')
        ax2.set_title("Harris Corner Detection")
        ax2.set_xlabel("Threshold: "+str(args.thresholdH)+"\nAlpha: "+str(args.alpha))

    elif args.method.lower()=="shitomasi":
        fig, (ax1, ax2) = plt.subplots(1,2)
        ax1.imshow(original)
        ax1.set_title("Input Image")
        ax2.imshow(original)
        ax2.plot(coordinates_shi[0],coordinates_shi[1],'.', color='firebrick')
        ax2.set_title("Shi-Tomasi Corner Detection")
        ax2.set_xlabel("Threshold: "+str(args.thresholdS))
    else:
        parser.print_help()
        print("Please enter the appropriate option")

    try:
        plt.show()
    except :
        pass
