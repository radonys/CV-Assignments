## CV Assignment - 2
### Q2 Report

***
#### Usage:
```python
usage: Q2.py [-h] [--confusion {0,1}] [--train_data_path TRAIN_DATA_PATH]
             [--test_data_path TEST_DATA_PATH]
             [--train_label_path TRAIN_LABEL_PATH]
             [--test_label_path TEST_LABEL_PATH] [--random_state RANDOM_STATE]
             [--max_iter MAX_ITER] [--C C] [--tol TOL]

Classification Using Multiclass 1 vs all SVM (linear Kernel) and Bag Of Visual
Words

optional arguments:
  -h, --help            show this help message and exit
  --confusion {0,1}     Generate and save confusion matrix
  --train_data_path TRAIN_DATA_PATH
                        Path to train folder/pkl
  --test_data_path TEST_DATA_PATH
                        Path to test folder/pkl
  --train_label_path TRAIN_LABEL_PATH
                        Path to train label file/pkl
  --test_label_path TEST_LABEL_PATH
                        Path to test labels file/pkl
  --random_state RANDOM_STATE
                        Random initialization. useful for result reproduction.
                        Use None for random.
  --max_iter MAX_ITER   Maximum iteration for SVM to be trained.
  --C C                 Outlier Penalty
  --tol TOL             Minimum error to converge in.

```
#### Result:
Best Accuracy: **60.125%**


#### Conditions:
```
1. random_state = 0
2. Outlier C = 0.001
3. max_iter = 10000
4. Tolerance tol = 0.01
```
#### Stats:<br>

Number of Clusters | Accuracy
--- | ---
10 | 40.25%
20 | 50.375%
30 | 50.125%
40 | 54.25%
50 | 53.75%
60 | 53.375%
70 | 57.125%
80 | 56.75%
90 | 58.625%
**100** | **60.125%**

#### Confusion Matrix Image:

![confusion_matrix](confusion_matrix_100.png)
