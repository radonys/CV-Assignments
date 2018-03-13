import numpy as np
import os
import heapq
import pickle
from sklearn.cluster import KMeans
# class cus_heap():
#     def __init__(self,data=[]):
#         self.data = data
#         self.heapify()
#     def heapify(self):
#         heapq.heapify(self.data)
#     def push(self,ele):
#         heap.heappush(self.data,ele)
#     def pop(self):
#         return heap.heappop(self.data)
#     def nsmallest(self,n):
#         return heap.nsmallest(self.data,n)
#     def reinit():
#         self.data = []
N = 9
labels_training = np.loadtxt(open("../Data/train_labels.csv"),delimiter=',')
labels_testing = np.loadtxt(open("../Data/test_labels.csv"),delimiter=',')
acc = 0
count = 0

def knn(testing,training,k,label_test,label_train):
    acc = 0.0
    for index,feature in enumerate(testing):
        distance = np.sqrt(np.sum(np.square(np.subtract(training,feature)),1))
        predictions = [ i[1] for i in heapq.nsmallest(k,list(np.append(distance,labels_training.T,1))) ]
        if max(set(predictions),key=predictions.count) == label_test[index]:
            acc+=1
    print("Accuracy: {}%".format(100*acc/index))

def kmeans(K):
    main_data = np.array([])
    print("Appending features")
    for i in os.listdir("../Data/train_sift_features"):
        if len(main_data):
            main_data = np.vstack((main_data,np.loadtxt(open("../Data/train_sift_features/"+i),delimiter=",")[:,4:]))
        else:
            main_data = np.loadtxt(open("../Data/train_sift_features/"+i),delimiter=",")[:,4:]
    print(main_data.shape)
    print("Starting Clustering")
    t = KMeans(n_clusters=K).fit(main_data)
    pickle.dump(t,open("k-means.pkl","wb"))

def training_features(kmeans):
    data = []
    labels_features = []
    for i in os.listdir("../Data/train_sift_features"):
        raw_data = np.loadtxt(open("../Data/train_sift_features/"+i),delimiter=",")[:,4:]
        labels_features.append(labels_training[int(i.split("_")[0])-1])
        image_features = [0 for i in range(kmeans.n_clusters)]
        while len(raw_data):
            y=raw_data[0]
            raw_data = np.delete(raw_data,0,0)
            image_features[np.argmin(np.sum(np.square(kmeans.cluster_centers_-y),1))]+=1
        data.append(image_features)
    pickle.dump(data,open("training_features.pkl","wb"))
    pickle.dump(labels_features,open("labels_training_features.pkl","wb"))

def test_features(kmeans):
    data = []
    labels_features = []
    for i in os.listdir("../Data/test_sift_features"):
        raw_data = np.loadtxt(open("../Data/test_sift_features/"+i),delimiter=",")[:,4:]
        labels_features.append(labels_testing[int(i.split("_")[0])-1])
        image_features = [0 for i in range(kmeans.n_clusters)]
        while len(raw_data):
            y=raw_data[0]
            raw_data = np.delete(raw_data,0,0)
            image_features[np.argmin(np.sum(np.square(kmeans.cluster_centers_-y),1))]+=1
        data.append(image_features)
    pickle.dump(data,open("test_features.pkl","wb"))
    pickle.dump(labels_features,open("labels_test_features.pkl","wb"))
if __name__ == "__main__":
    label_test = np.array(pickle.load(open("labels_test_features.pkl","rb")))
    label_train = np.array(pickle.load(open("labels_training_features.pkl","rb")))
    training = pickle.load(open("training_features.pkl","rb"))
    testing = pickle.load(open("test_features.pkl","rb"))
    print("Started KNN")
    knn(testing,training,9,label_test,label_train)
    # km = pickle.load(open("k-means.pkl","rb"))
    # print("Generating Test Features")
    # test_features(km)
    # print("Generating Training Features")
    # training_features(km)
