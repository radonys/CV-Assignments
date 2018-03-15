from sklearn.svm import LinearSVC
import pickle
import time
import numpy as np
from plotConfusionMatrix import plotConfusionMatrixFunction
import argparse

def multiClassSVM(test, train, label_test, label_train,random_state,C,max_iter,tol,image_name="confusion_matrix.png", print_image=False,):
    clf = LinearSVC(multi_class='ovr',random_state=random_state,C=C,max_iter=max_iter,tol=tol)
    print(clf)
    start = time.process_time()
    clf.fit(train, label_train)
    print("Training time: {} sec".format(time.process_time()-start))
    acc = 0.0
    zero_matrix = np.zeros((8,8)).tolist()
    start = time.process_time()
    for index,i in enumerate(test):
        predicted = int(clf.predict(np.array(i).reshape(1,-1))[0])
        original = int(label_test[index])
        if predicted == original:
            acc+=1
        zero_matrix[predicted-1][original-1]+=1
    print("Testing time: {} sec".format(time.process_time()-start))
    print("Accuracy: {}%".format(acc*100/len(test)))
    if print_image:
        plotConfusionMatrixFunction(zero_matrix, image_name)

if __name__ == "__main__":
    try:
        parser = argparse.ArgumentParser(description='Classification Using Multiclass 1 vs all SVM (linear Kernel) and Bag Of Visual Words',prog='Q2.py')
        parser.add_argument("--confusion",type=int,choices=[0,1],help="Generate and save confusion matrix",default=0)
        parser.add_argument("--train_data_path",type=str,help="Path to train folder/pkl",default="../features/training_features80.pkl")
        parser.add_argument("--test_data_path",type=str,help="Path to test folder/pkl",default="../features/test_features80.pkl")
        parser.add_argument("--train_label_path",type=str,help="Path to train label file/pkl",default="../features/labels_training_features_80.pkl")
        parser.add_argument("--test_label_path",type=str,help="Path to test labels file/pkl",default="../features/labels_test_features80.pkl")
        parser.add_argument("--random_state",type=int,help="Random initialization. useful for result reproduction. Use None for random.",default=0)
        parser.add_argument("--max_iter",type=int,help="Maximum iteration for SVM to be trained.",default=10000)
        parser.add_argument("--C",type=float,help="Outlier Penalty",default=0.001)
        parser.add_argument("--tol",type=float,help="Minimum error to converge in.",default=0.01)
        args = parser.parse_args()
        test = pickle.load(open(args.test_data_path,"rb"))
        train = pickle.load(open(args.train_data_path,"rb"))
        label_test = pickle.load(open(args.test_label_path,"rb"))
        label_train = pickle.load(open(args.train_label_path,"rb"))
        print_image = args.confusion
        multiClassSVM(test, train, label_test, label_train, print_image=print_image,random_state=args.random_state,C=args.C,max_iter=args.max_iter,tol=args.tol)
    except FileNotFoundError as e:
        print(e)
        parser.print_help()
