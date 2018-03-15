# Question 4 - Assignment 2

## About the Code

Finetuning AlexNet's and VGG16 layers where the classes are reduced from 1000 to the number of classes in dataset(8) in the classifier layer.

## Installing Dependencies

1. Install PyTorch by following the instructions given on the link [here](http://pytorch.org/).
2. Run ```pip3 install torchvision tensorflow==1.4 tensorboardX``` in command line.

## Running the Code

#### For AlexNet
1. First, get the dataset in two folders: train and test by modifying paths in [data_process.py](https://github.com/radonys/CV-Assignments/blob/master/Assignment-2/Q3/data_process.py).
Then, execute the command ```python3 data_process.py```.
2. Then modify the dataset path in [alexnet.py](https://github.com/radonys/CV-Assignments/blob/master/Assignment-2/Q3/alexnet.py), where the newly created folders are saved.
3. Then run ```python3 alexnet.py```.

#### For VGG16

1. Repeat the 2 and 3 steps as mentioned for AlexNet with the following file: [vgg16.py](https://github.com/radonys/CV-Assignments/blob/master/Assignment-2/Q4/vgg16.py).

## Visualizing the Results

To visualize the current results, open a terminal session, and run the following command:
```tensorboard --logdir runs```

## Obtained Results

#### For AlexNet (Epochs = 150)

1. Training Accuracy = **92.53%**
2. Training Loss = **0.26**
3. Testing Accuracy = **92.5**

#### For VGG16 (Epochs = 16)

1. Training Accuracy = **89.63%**
2. Training Loss = **0.34**
3. Testing Accuracy = **88.38**

## Graphs

#### For AlexNet (Epochs = 150)

![Train Accuracy](https://github.com/radonys/CV-Assignments/blob/master/Assignment-2/Q4/alexnettrainacc.png "Train Accuracy")
![Train Loss](https://github.com/radonys/CV-Assignments/blob/master/Assignment-2/Q4/alexnettrainloss.png "Train Loss")
![Test Accuracy](https://github.com/radonys/CV-Assignments/blob/master/Assignment-2/Q4/alexnettestacc.png "Test Accuracy")

#### For VGG16 (Epochs = 16)

![Train Accuracy](https://github.com/radonys/CV-Assignments/blob/master/Assignment-2/Q4/vggtrainacc.png "Train Accuracy")
![Train Loss](https://github.com/radonys/CV-Assignments/blob/master/Assignment-2/Q4/vggtrainloss.png "Train Loss")
![Test Accuracy](https://github.com/radonys/CV-Assignments/blob/master/Assignment-2/Q4/vggtestacc.png "Test Accuracy")
