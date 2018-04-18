import cv2
import numpy as np
import matplotlib.pyplot as plt
import sys
import random
from scipy.sparse.linalg import svds
from sklearn import linear_model
Detector = cv2.xfeatures2d.SIFT_create()
def SIFT(image):
    return Detector.detectAndCompute(image,None)

def matches(feature1,feature2,kp1,kp2):
    kps1 = []
    features1 = []
    kps2 = []
    features2 = []
    for k,i in enumerate(feature2):
        ind = np.argsort(np.linalg.norm(np.subtract(feature1,i),axis=1))[:2]
        if np.linalg.norm(i-feature1[ind[0]])< 0.7 * np.linalg.norm(i-feature1[ind[1]]): #ratio : 0.7-0.8
            kps2.append(kp2[k])
            features2.append(i)
            kps1.append(kp1[ind[0]])
            features1.append(feature1[ind[0]])
    return ((kps1,features1),(kps2,features2))

def merge_images():
    pass
if __name__ == '__main__':
    img1 = cv2.imread('uttower_left.JPG')[...,::-1]
    img2 = cv2.imread('uttower_right.JPG')[...,::-1]
    kp1,feature1 = SIFT(img1)
    kp2,feature2 = SIFT(img2)
    ((kps1,features1),(kps2,features2)) = matches(feature1,feature2,kp1,kp2)
    #merging 2 images to show matches
    (hA, wA) = img1.shape[:2]
    (hB, wB) = img2.shape[:2]
    vis = np.zeros((max(hA, hB), wA + wB, 3), dtype="uint8")
    vis[0:hA, 0:wA] = img1
    vis[0:hB, wA:] = img2
    for i in range(len(features2)):
        ptA = (int(kps1[i].pt[0]), int(kps1[i].pt[1]))
        ptB = (int(kps2[i].pt[0]) + wA, int(kps2[i].pt[1]))
        cv2.line(vis, ptA, ptB, (random.randint(0,255), random.randint(0,255), random.randint(0,255)), 1)
    plt.title("Matches")
    plt.imshow(vis)
    kps1 = np.array(kps1)
    kps2 = np.array(kps2)
    #calculating homography
    kp1_coordinates = np.array([ [i.pt[0],i.pt[1],1]for i in kps1])
    kp2_coordinates = np.array([ [i.pt[0],i.pt[1],1]for i in kps2])
    # ransac = linear_model.RANSACRegressor()
    # ransac.fit(kp2_coordinates, kp1_coordinates)
    # print(ransac.estimator_.coef_)
    if len(kps1)>=4:
        best_h = np.array([])
        best_count = -1
        for i in range(len(kps1)):
            ind = np.array(random.sample(list(range(len(kps1))),4))
            x_p = np.array([[v.pt[0],v.pt[1]] for v in kps1[ind]])
            x = np.array([[v.pt[0],v.pt[1]] for v in kps2[ind]])
            A = np.array([])
            for j in range(4):
                A = np.vstack([A,
                [[-x[j,0],-x[j,1],-1,0,0,0,x[j,0]*x_p[j,0],x[j,1]*x_p[j,0],x_p[j,0]],
                [0,0,0,-x[j,0],-x[j,1],-1,x[j,0]*x_p[j,1],x[j,1]*x_p[j,1],x_p[j,1]]]]
                ) if A.size else np.array([[-x[j,0],-x[j,1],-1,0,0,0,x[j,0]*x_p[j,0],x[j,1]*x_p[j,0],x_p[j,0]],[0,0,0,-x[j,0],-x[j,1],-1,x[j,0]*x_p[j,1],x[j,1]*x_p[j,1],x_p[j,1]]])
            u,s,vh = np.linalg.svd(A)
            xtemp = np.hstack([x_p,np.ones((x.shape[0],1))])
            xori = np.hstack([x,np.ones((x_p.shape[0],1))])
            L = vh[-1,:]
            h = L.reshape((3,3))
            homo_x = np.matmul(h,kp2_coordinates.T)
            k = np.linalg.norm(kp1_coordinates-homo_x.T,axis=1)<3
            if best_count<np.count_nonzero(k):
                best_count = np.count_nonzero(k)
                best_h = h
    # print(best_count)
    result = cv2.warpPerspective(img2,best_h,
    	(img1.shape[1] + img2.shape[1], img1.shape[0]))
    result[0:img1.shape[0],0:img2.shape[1]] = img1

    H,mask = cv2.findHomography(kp2_coordinates, kp1_coordinates, cv2.RANSAC,5.0)
    result1 = cv2.warpPerspective(img2,H,
    	(img1.shape[1] + img2.shape[1], img1.shape[0]))
    result1[0:img1.shape[0],0:img2.shape[1]] = img1
    plt.figure()
    plt.title("Merged Image: Own implementation")
    plt.imshow(result)
    plt.figure()
    plt.title("Merged Image: Using CV2 homography")
    plt.imshow(result1)




    plt.show()
