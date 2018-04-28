import cv2
import numpy as np
import numpy
import matplotlib.pyplot as plt
import sys
import random
NN_THRESHOLD = 0.7
PIXEL_ERROR_THRESHOLD = 3
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
        if np.linalg.norm(i-feature1[ind[0]])< NN_THRESHOLD * np.linalg.norm(i-feature1[ind[1]]): #ratio : 0.7-0.8
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
    plt.imsave("Matches.jpg",vis)
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
            x_p = np.array([[v.pt[0],v.pt[1],1] for v in kps1[ind]])
            x = np.array([[v.pt[0],v.pt[1],1] for v in kps2[ind]])
            A = np.array([])
            fp = x
            tp = x_p
            m = numpy.mean(fp[:2], axis=1)
            maxstd = numpy.max(numpy.std(fp[:2], axis=1)) + 1e-9
            C1 = numpy.diag([1/maxstd, 1/maxstd, 1])
            C1[0, 2] = -m[0] / maxstd
            C1[1, 2] = -m[1] / maxstd
            fp = numpy.dot(C1, fp.T)

            # -to
            m = numpy.mean(tp[:2], axis=1)
            maxstd = numpy.max(numpy.std(tp[:2], axis=1)) + 1e-9
            C2 = numpy.diag([1/maxstd, 1/maxstd, 1])
            C2[0, 2] = -m[0] / maxstd
            C2[1, 2] = -m[1] / maxstd
            tp = numpy.dot(C2, tp.T)

            correspondences_count = fp.shape[1]
            A = numpy.zeros((2 * correspondences_count, 9))
            for i in range(correspondences_count):
                A[2 * i    ] = [-fp[0][i], -fp[1][i], -1, 0, 0, 0,
                tp[0][i]  * fp[0][i], tp[0][i] * fp[1][i], tp[0][i]]
                A[2 * i + 1] = [0, 0, 0, -fp[0][i], -fp[1][i], -1,
                tp[1][i]  * fp[0][i], tp[1][i] * fp[1][i], tp[1][i]]

            U, S, V = numpy.linalg.svd(A)
            H = V[8].reshape((3, 3))

            H = numpy.dot(numpy.linalg.inv(C2), numpy.dot(H, C1))
            h = H/H[2,2]
            homo_x = np.matmul(h,kp2_coordinates.T)
            k = np.linalg.norm(kp1_coordinates-homo_x.T,axis=1)<PIXEL_ERROR_THRESHOLD
            if best_count<np.count_nonzero(k):
                best_count = np.count_nonzero(k)
                best_h = h
            # return H / H[2, 2]



    print(best_count)
    result = cv2.warpPerspective(img2,best_h,
    	(img1.shape[1] + img2.shape[1], img1.shape[0]))
    plt.imsave("Homography.jpg",result)
    result[0:img1.shape[0],0:img2.shape[1]] = img1
    plt.imsave("Result.jpg",result)

    H,mask = cv2.findHomography(kp2_coordinates, kp1_coordinates, cv2.RANSAC,PIXEL_ERROR_THRESHOLD)
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
