import cv2
import csv
import glob
import os
from natsort import natsorted, ns

def csvparse(csvfile):

    with open(csvfile,'r') as classlist:

        classreader = csv.reader(classlist)
        for row in classreader:
            print(row)
            return row

def write_images(data_directory,class_list,out_dir):

    image_list = glob.glob(data_directory + '/*.jpg')

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    image_list = natsorted(image_list, alg = ns.IGNORECASE)
    
    for i in range(0,len(image_list)):
        
        filepath_old = image_list[i]
        image = cv2.imread(filepath_old)
        
        filepath_new = out_dir
        temp = filepath_old.split('/')
        filepath_new = filepath_new + '/' + class_list[i]
        if not os.path.exists(filepath_new):
            os.makedirs(filepath_new)
        filepath_new = filepath_new + '/' + temp[len(temp)-1]
        print("Writing at: " + filepath_new)
        cv2.imwrite(filepath_new,image)

train_data_dir = '/Users/yashsrivastava/Desktop/hw2_data/train'
train_out = '/Users/yashsrivastava/Desktop/hw2_data/trainf'
train_csv = '/Users/yashsrivastava/Desktop/hw2_data/train_labels.csv'

test_data_dir = '/Users/yashsrivastava/Desktop/hw2_data/test'
test_out = '/Users/yashsrivastava/Desktop/hw2_data/testf'
test_csv = '/Users/yashsrivastava/Desktop/hw2_data/test_labels.csv'

print("Writing Train Images..")
write_images(train_data_dir,csvparse(train_csv),train_out)

'''
print("Writing Test Images..")
write_images(test_data_dir,csvparse(test_csv),test_out)
'''

print("Process Done.")

