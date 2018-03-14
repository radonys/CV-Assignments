from sklearn.svm import LinearSVC
import pickle
import time
import numpy as np
from plotConfusionMatrix import plotConfusionMatrixFunction

def multiClassSVM(test, train, label_test, label_train, print_image=False,image_name="confusion_matrix.png"):
    # for k in range(10,110,10):
    #     print("Cluster size:{}".format(k))
    #     label_test = pickle.load(open('../../../features/labels_test_features{}.pkl'.format(k),'rb'))
    #     label_train = pickle.load(open('../../../features/labels_training_features_{}.pkl'.format(k), 'rb'))
    #     test = pickle.load(open('../../../features/test_features{}.pkl'.format(k), 'rb'))
    #     train = pickle.load(open('../../../features/training_features{}.pkl'.format(k), 'rb'))
    start = time.process_time()
    clf = LinearSVC(multi_class='ovr',random_state=0,C=0.001,max_iter=10000,tol=0.01)
    print(len(train))
    print(len(label_train))
    clf.fit(train, label_train)
    print("time: {}sec".format(time.process_time()-start))
    acc = 0.0
    zero_matrix = np.zeros((8,8)).tolist()

    for index,i in enumerate(test):
        predicted = int(clf.predict(np.array(i).reshape(1,-1))[0])
        original = int(label_test[index])
        if predicted == original:
            acc+=1
        zero_matrix[predicted-1][original-1]+=1
    print("Accuracy: {}%".format(acc*100/len(test)))

    #print image
    if print_image:
        plotConfusionMatrixFunction(zero_matrix, image_name)

if __name__ == "__main__":
    parser.add_argument('--k',type=int,help="K for KMeans, KNN or Kfolds (compulsary)")
    parser.add_argument("--kmeans_cluster",type=str,help="Path cluster pkl file",default="50-means.pkl")
    parser.add_argument("--train_data_path",type=str,help="Path to train folder/pkl",default="../Data/train_sift_features")
    parser.add_argument("--test_data_path",type=str,help="Path to test folder/pkl",default="../Data/test_sift_features")
    parser.add_argument("--train_label_path",type=str,help="Path to train label file/pkl",default="../Data/train_labels.csv")
    parser.add_argument("--test_label_path",type=str,help="Path to test labels file/pkl",default="../Data/test_labels.csv")
    args = parser.parse_args()
