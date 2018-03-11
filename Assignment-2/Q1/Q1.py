import numpy as np
import os
import heapq
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
for j in os.listdir("../Data/test_sift_features"):
    for i in os.listdir("../Data/train_sift_features"):
        f1 = np.loadtxt(open("../Data/train_sift_features/"+i),delimiter=',')
        f2 = np.loadtxt(open("../Data/test_sift_features/"+j),delimiter=',')
        print(f1.shape,f2.shape)
        heap.push((Edistance(f1.flatten(),f2.flatten()),int(i.split("_")[0])-1))
    res = [labels_training(i[1]) for i in heap.nsmallest(N)]
    if max(set(res),key=res.count)==labels_testing[int(j.split("_")[0])]:
        print("match: {}".format(labels_testing[int(j.split("_")[0])]))
        acc+=1
print("acc:{}",acc/len(labels_testing))
