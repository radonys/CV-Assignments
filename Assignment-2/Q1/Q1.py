import numpy as np
import os
import heapq
import pickle
from sklearn.cluster import KMeans
from sklearn.model_selection import KFold
import argparse
import sys
import os
import time
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


def cross_validation(arr,n_split,labels,k_start=1,k_end=101):
    kf = KFold(n_splits=n_split)
    results = []
    for test,train in kf.split(arr):
        test_split = arr[test]
        train_split = arr[train]
        label_test_split = labels[test]
        label_train_split = labels[train]
        #print(test_split.shape,train_split.shape)
        for i in range(k_start,k_end+1,2):
            results.append((knn(test_split,train_split,i,label_test_split,label_train_split,DEBUG=False),i))
    print(max(results))
    return max(results)

def knn(testing,training,k,label_test,label_train,DEBUG=True):
    acc = 0.0
    for index,feature in enumerate(testing):
        distance = np.sqrt(np.sum(np.square(training-feature),1).reshape(-1,1))
        #print(label_train.shape)
        predictions = [ i[1] for i in heapq.nsmallest(k,np.append(distance,label_train.reshape(-1,1),1).tolist()) ]
        #print(predictions, max(set(predictions),key=predictions.count),label_test[index],index)
        if max(set(predictions),key=predictions.count) == label_test[index]:
            acc+=1
    if DEBUG:
        print("Accuracy: {}%".format(100*acc/index))
    return(100*acc/index)

def kmeans(K,path):
    main_data = np.array([])
    print("Appending features")
    for i in os.listdir(path):
        if len(main_data):
            main_data = np.vstack((main_data,np.loadtxt(open(os.path.join(path,i)),delimiter=",")[:,4:]))
        else:
            main_data = np.loadtxt(open(os.path.join(path,i)),delimiter=",")[:,4:]
    print(main_data.shape)
    print("Starting Clustering")
    t = KMeans(n_clusters=K).fit(main_data)
    pickle.dump(t,open("{}-means.pkl".format(K),"wb"))
    return t

def training_features(kmeans,path,labels_training):
    data = []
    labels_features = []
    for i in os.listdir(path):
        raw_data = np.loadtxt(open(os.path.join(path,i)),delimiter=",")[:,4:]
        labels_features.append(labels_training[int(i.split("_")[0])-1])
        image_features = [0 for i in range(kmeans.n_clusters)]
        while len(raw_data):
            y=raw_data[0]
            raw_data = np.delete(raw_data,0,0)
            image_features[np.argmin(np.sqrt(np.sum(np.square(kmeans.cluster_centers_-y),1)))]+=1
        data.append(image_features)
    pickle.dump(data,open("training_features.pkl","wb"))
    pickle.dump(labels_features,open("labels_training_features.pkl","wb"))

def test_features(kmeans,path,labels_testing):
    data = []
    labels_features = []
    for i in os.listdir(path):
        raw_data = np.loadtxt(open(os.path.join(path,i)),delimiter=",")[:,4:]
        labels_features.append(labels_testing[int(i.split("_")[0])-1])
        image_features = [0 for i in range(kmeans.n_clusters)]
        while len(raw_data):
            y=raw_data[0]
            raw_data = np.delete(raw_data,0,0)
            image_features[np.argmin(np.sqrt(np.sum(np.square(kmeans.cluster_centers_-y),1)))]+=1
        data.append(image_features)
    pickle.dump(data,open("test_features{}.pkl".format(kmeans.cluster_centers_[0].shape[0]),"wb"))
    pickle.dump(labels_features,open("labels_test_features{}.pkl".format(kmeans.cluster_centers_[0].shape[0]),"wb"))
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Classification Using Knn and Bag Of Visual Words',prog='Q1.py')
    parser.add_argument('--mode', type=int,help="""Mode for running code:\n
    1)Kmeans bag of words Generation and feature Generation\n
    2)Cross validation in Kfolds\n
    3)Classification using KNN
    """,choices=[1,2,3],default=1)
    parser.add_argument('--k',type=int,help="K for KMeans, KNN or Kfolds (compulsary)")
    parser.add_argument("--train_data_path",type=str,help="Path to train folder/pkl",default="../Data/train_sift_features")
    parser.add_argument("--test_data_path",type=str,help="Path to test folder/pkl",default="../Data/test_sift_features")
    parser.add_argument("--train_label_path",type=str,help="Path to train label file/pkl",default="../Data/train_labels.csv")
    parser.add_argument("--test_label_path",type=str,help="Path to test labels file/pkl",default="../Data/test_labels.csv")
    args = parser.parse_args()
    if args.k==None:
        print("Please provide K value")
        parser.print_help()
        sys.exit()

    if args.mode == 1:
        if args.train_data_path.endswith(".pkl") and args.test_data_path.endswith(".pkl") and args.train_label_path.endswith(".pkl")  and args.test_label_path.endswith(".pkl"):
            print("Please provide the original image path for the Generation mode")
            args.print_help()
            sys.exit()
        else:
            start = time.process_time()
            centers = kmeans(args.k,args.train_data_path)
            print("K-Means clustering done. Time Taken: {} secs".format(time.process_time()-start))
            print("Generating Features")
            labels_training = np.loadtxt(open(args.train_label_path),delimiter=',')
            labels_testing = np.loadtxt(open(args.test_label_path),delimiter=',')
            start = time.process_time()
            training_features(centers,args.train_data_path,labels_training)
            test_features(centers,args.test_data_path,labels_testing)
            print("Feature Generation done. Time Taken: {} secs ".format(time.process_time()-start))
    elif args.mode == 2:
        if args.train_data_path.endswith(".pkl") and args.train_label_path.endswith(".pkl"):
            label_test = np.array(pickle.load(open(args.test_label_path,"rb")))
            label_train = np.array(pickle.load(open(args.train_label_path,"rb")))
            training = np.array(pickle.load(open(args.train_data_path,"rb")))
            testing = np.array(pickle.load(open(args.test_label_path,"rb")))
            cross_validation(training,args.k,label_train)
        else:
            print("Please provide the pkl files for cross validation")
            args.print_help()
            sys.exit()
    elif args.mode == 3:
        if args.train_data_path.endswith(".pkl") and args.train_label_path.endswith(".pkl"):
            label_test = np.array(pickle.load(open(args.test_label_path,"rb")))
            label_train = np.array(pickle.load(open(args.train_label_path,"rb")))
            training = np.array(pickle.load(open(args.train_data_path,"rb")))
            testing = np.array(pickle.load(open(args.test_label_path,"rb")))
            knn(training,args.k,label_train)
        else:
            print("Please provide the pkl files for knn")
            args.print_help()
            sys.exit()
    else:
        print("Please provide proper options")
        args.print_help()
        sys.exit()
    # label_test = np.array(pickle.load(open("labels_test_features.pkl","rb")))
    # label_train = np.array(pickle.load(open("labels_training_features.pkl","rb")))
    # training = np.array(pickle.load(open("training_features.pkl","rb")))
    # testing = np.array(pickle.load(open("test_features.pkl","rb")))
    # print(training.shape)
    # print(testing.shape)
    # print("Started Knn")
    # knn(testing,training,43,label_test,label_train)
    # cross_validation(training,5,label_train)
    # km = pickle.load(open("k-means.pkl","rb"))
    # print("Generating Test Features")
    # test_features(km)
    # print("Generating Training Features")
    # training_features(km)
