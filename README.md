# Artificial Inteligence Assignment

In this exercise, available implementations of Naive Bayes and Decision Tree are used to compare the performance of these algorithms on different datasets.
Each datasets have size n of at least 1000 examples. The performance of the two algorithms on each datasets, are compared 
measuring the generalization error as the number of examples increases. For this purpose, the average and standard deviation of 
the error on test set are reported on a Learning Curve, plotted by sampling a certain number of training sets of size m < n, varying
m in logarithmic scale between 10% and 50% of n (reserving the remaining 50% for the test set). 

## Getting Start

This script was written using Python 3.6. 

### Prerequisites

Libraries which need to be installed for its use, are:

    - sklearn
    - numpy
    - scipy
    - matplotlib
    
### Usage

When you run the script, the list of avaiable datasets is shows. The number beside the name is to be entered to start processing on that dataset (give 0 to finish executing the script). NOTE: if one or more datasets is not already installed, the script needs an internet connection for download. For the first two datasets (20newsgroups and Reuters-21578), a sub-menu of choice appears to use or not, the TfidfTransformer function.
For the MNIST dataset, a sub-menu of choice appears to use the entire dataset size or not. 
During execution, the name of the current classifier appears, while, once finished, you will be asked if you want to save the .png image in a default directory, (current_directory/Plotted) or not (in which case, the script ends). The resulting graph is then shown, the figure is automatically saved, the execution time is shown and then it returns to the selection of the dataset.

## Output Examples

You can find every output in the folder Plotted.

![20 News Groups without TfidfTransformer](../master/Plotted/20Newsgroups0.png)

![MNIST Orignal complete size](../master/Plotted/MNISTOriginal0.png)

## Conclusions

There's also a pdf with mine conlcusions about this comparision (only Italian avaiable) ![AI Reporting](../master/AI Reporting.pdf)
    
## Authors

* **Federico Palai** - [my github](https://github.com/palai103)
    
   
