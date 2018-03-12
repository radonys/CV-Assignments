import numpy as np
import os
import heapq
import pickle
from sklearn.cluster import KMeans
class cus_heap():
    def __init__(self,data=[]):
        self.data = data
        self.heapify()
    def heapify(self):
        heapq.heapify(self.data)
    def push(self,ele):
        heap.heappush(self.data,ele)
    def pop(self):
        return heap.heappop(self.data)
    def nsmallest(self,n):
        return heap.nsmallest(self.data,n)
    def reinit():
        self.data = []
N = 9
labels_training = np.loadtxt(open("../Data/train_labels.csv"),delimiter=',')
labels_testing = np.loadtxt(open("../Data/train_labels.csv"),delimiter=',')
acc = 0
count = 0

def kmeans(K):
    main_data = np.array([])
    for i in os.listdir("../Data/train_sift_features"):
        if len(main_data):
            main_data = np.vstack((main_data,np.loadtxt(open("../Data/train_sift_features/"+i),delimiter=",")))
        else:
            main_data = np.loadtxt(open("../Data/train_sift_features/"+i),delimiter=",")
    print(main_data.shape)
    t = KMeans(n_clusters=50).fit(main_data)
    pickle.dump(t,open("k-means.pkl","wb"))

def training_features(kmeans):
    data = []
    labels_features = []
    for i in os.listdir("../Data/train_sift_features"):
        raw_data = np.loadtxt(open("../Data/train_sift_features/"+i),delimiter=",")
        labels_features.append(labels_training[i.split("_")[0]-1])
        image_features = [0 for i in range(kmeans.n_clusters)]
        while len(raw_data):
            image_features[np.argmin(np.sum(np.square(kmeans.cluster_centers_-np.delete(raw_data,0,0),1)))]+=1
        data.append(image_features)
    pickle.dump(data,open("training_features.pkl","wb"))
    pickle.dump(labels_features,open("labels_training_features.pkl","wb"))

def test_features(kmeans):
    data = []
    labels_features = []
    for i in os.listdir("../Data/test_sift_features"):
        raw_data = np.loadtxt(open("../Data/test_sift_features/"+i),delimiter=",")
        labels_features.append(labels_training[i.split("_")[0]-1])
        image_features = [0 for i in range(kmeans.n_clusters)]
        while len(raw_data):
            image_features[np.argmin(np.sum(np.square(kmeans.cluster_centers_-np.delete(raw_data,0,0),1)))]+=1
        data.append(image_features)
    pickle.dump(data,open("test_features.pkl","wb"))
    pickle.dump(labels_features,open("labels_test_features.pkl","wb"))

km = pickle.load(open("k-means.pkl","rb"))
print("Generating Test Features")
test_features(km)
print("Generating Training Features")
training_features(km)
