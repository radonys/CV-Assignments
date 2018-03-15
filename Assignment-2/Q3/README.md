# Question 3 - Assignment 2

## About the Code

Finetuning AlexNet's last classifier layer where the classes are reduced from 1000 to the number of classes in dataset(8).

## Installing Dependencies

1. Install PyTorch by following the instructions given on the link [here](http://pytorch.org/).
2. Run ```pip3 install torchvision tensorflow==1.4 tensorboardX``` in command line.

## Running the Code

1. First, get the dataset in two folders: train and test by modifying paths in [data_process.py](https://github.com/radonys/CV-Assignments/blob/master/Assignment-2/Q3/data_process.py).
Then, execute the command ```python3 data_process.py```.
2. Then modify the dataset path in [alexnet.py](https://github.com/radonys/CV-Assignments/blob/master/Assignment-2/Q3/alexnet.py), where the newly created folders are saved.
3. Then run ```python3 alexnet.py```.

## Visualizing the Results

To visualize the current results, open a terminal session, and run the following command:
```tensorboard --logdir runs```

## Obtained Results

1. Training Accuracy = **92.69%**
2. Training Loss = **0.27**
3. Testing Accuracy = **92.5**

## Graphs

![Train Accuracy](https://github.com/radonys/CV-Assignments/blob/master/Assignment-2/Q3/trainacc.png "Train Accuracy")
![Train Loss](https://github.com/radonys/CV-Assignments/blob/master/Assignment-2/Q3/trainloss.png "Train Loss")
![Test Accuracy](https://github.com/radonys/CV-Assignments/blob/master/Assignment-2/Q3/testacc.png "Test Accuracy")
