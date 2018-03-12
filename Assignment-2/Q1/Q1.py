import numpy as np
import os
import heapq
import pickle
from sklearn.cluster import KMeans
def Edistance(v1,v2):
    return np.sum(np.subtract(v1,v2)**2)
N = 9
labels_training = np.loadtxt(open("../Data/train_labels.csv"),delimiter=',')
labels_testing = np.loadtxt(open("../Data/train_labels.csv"),delimiter=',')

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
acc = 0
count = 0
heap = cus_heap()
main_data = np.array([])
for i in os.listdir("../Data/train_sift_features"):
    if len(main_data):
        main_data = np.vstack((main_data,np.loadtxt(open("../Data/train_sift_features/"+i),delimiter=",")))
    else:
        main_data = np.loadtxt(open("../Data/train_sift_features/"+i),delimiter=",")
print(main_data.shape)
t = KMeans(n_clusters=50).fit(main_data)
pickle.dump(t,open("k-means.pkl","wb"))
print(t.cluster_centers_)
print("acc:{}",acc/len(labels_testing))
