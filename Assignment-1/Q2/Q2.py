import cv2 as cv
import numpy as np
from numpy.linalg import eig
#  Method
# ->Find hessian of matrix using laplacian
#
#
#
#
def eigenvals(matrix):
    return np.real(eig(matrix))
def readimg(path):
    return np.pad(cv.imread(path,0),((1,1),(1,1)),'constant',constant_values=0)
def shiTomasi(img):
    pass
def harris(img):
    pass
if __name__=="__main__":
    # img = readimg("")
