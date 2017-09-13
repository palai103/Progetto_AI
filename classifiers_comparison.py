import os
import os.path
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer

from reuters_parser import filter_documents, stream_reuters_documents


def manual_plot(X, y, title):
    """
    The follow function was taken from Scikitlearn documentation and slighty modified for our purpose
    ("http://scikit-learn.org/stable/auto_examples/linear_model/plot_sgd_comparison.html#sphx-glr-auto-examples-linear-model-plot-sgd-comparison-py")

    :param X: data
    :param y: target
    :param title: current dataset name
    :return: plotted graph
    """
    start = datetime.now()
    classifiers = [("Naive Bayes", MultinomialNB()), ("Decision Tree", DecisionTreeClassifier())]
    heldout = [0.9, 0.8, 0.7, 0.6, 0.5]
    xx = 1. - np.array(heldout)
    rounds = 20
    for name, clf in classifiers:
        print("training %s" % name)
        rng = np.random.RandomState(42)
        yy = []
        standard_deviation = []
        for i in xx:
            yy_ = []
            for r in range(rounds):
                X_train, X_test, y_train, y_test = \
                    train_test_split(X, y, test_size=0.5, train_size=i, random_state=rng)
                clf.fit(X_train, y_train)
                y_pred = clf.predict(X_test)
                yy_.append(1 - accuracy_score(y_true=y_test, y_pred=y_pred))
            yy.append(np.mean(yy_))
            standard_deviation.append(np.std(yy_))
        standard_deviation = np.array(standard_deviation)
        yy = np.array(yy)
        if name == "Naive Bayes":
            plt.plot(xx, yy, '.-', color="r", label=name)
            plt.fill_between(xx, yy - standard_deviation, yy + standard_deviation, alpha=0.2, color='r')
            plt.errorbar(xx, yy, yerr=standard_deviation, linestyle='None', color='r')
        elif name == "Decision Tree":
            plt.plot(xx, yy, '.-', color="g", label=name)
            plt.fill_between(xx, yy - standard_deviation, yy + standard_deviation, alpha=0.2, color='g')
            plt.errorbar(xx, yy, yerr=standard_deviation, linestyle='None', color='g')

    plt.grid()
    plt.title(title)
    plt.legend(loc="upper right")
    plt.xlabel("Proportion train")
    plt.ylabel("Test Error Rate")
    end = datetime.now() - start
    save_picture(title, end)
    plt.show()


def save_picture(dataset_name, execution_time):
    """
    :param dataset_name: current dataset name
    :param execution_time: total execution time
    :return: save the picture in the default directory
    """
    dataset_name = dataset_name.replace(" ", "")
    plotted_directory = os.path.join(os.path.curdir, "Plotted")
    if not os.path.exists(plotted_directory):
        os.makedirs(plotted_directory)
    default_directory = os.path.join(plotted_directory, dataset_name)

    which_directory = """Want to save in the default directory (~/Plotted)?
1 - Yes
0 - No \n"""
    choice = int(input(which_directory))
    if choice == 1:
        directory = default_directory
    elif choice == 0:
        print("Image not saved. \n")
        exit()

    image_number = 0
    while os.path.exists('{}{:d}.png'.format(directory, image_number)):
        image_number += 1
    plt.savefig('{}{:d}.png'.format(directory, image_number))
    print("Execution time: {}".format(execution_time))


def use_tfidf(X_train_counts):
    """
    :param X_train_counts: sparse representation of the counts
    :return: a collection of raw documents, converted to a matrix of TF-IDF features or the input sparse representation
    """
    tfidf = """Proceed without using Tfidf Transformation?
1 - Yes
0 - No \n"""
    choice = int(input(tfidf))
    if choice == 0:
        return TfidfTransformer().fit_transform(X_train_counts)
    elif choice == 1:
        return X_train_counts


"""============================================= DATASETS ============================================="""


def twenty_newsgroups():
    """
    ====== Twenty News Groups dataset ("http://scikit-learn.org/stable/datasets/twenty_newsgroups.html") ======
    """

    train_set = datasets.fetch_20newsgroups(subset='all', remove=('headers', 'quote', 'footers'))
    X_train_counts = CountVectorizer().fit_transform(train_set.data)
    X_train_tfidf = use_tfidf(X_train_counts)
    print('\n', X_train_tfidf.shape, '\n')

    # Set X, y
    X, y = X_train_tfidf, train_set.target

    # Plot
    manual_plot(X, y, title="20 Newsgroups")


def reuters():
    """
    ============== Reuters 21578 dataset ("http://scikit-learn.org/stable/auto_examples/applications/plot_out_of_core_classification.html") ==============
    """

    cont = u'{title}\n\n{body}'
    iterator = stream_reuters_documents()
    X, y = filter_documents(iterator, cont)
    X_train_counts = CountVectorizer().fit_transform(X)
    X_train_tfidf = use_tfidf(X_train_counts)

    # Set X, y
    X = X_train_tfidf
    print('\n', X.shape, '\n')

    # Plot
    manual_plot(X, y, title="Reuters - 21578")


def mnist():
    """
    ====== MNIST dataset ("http://mldata.org/repository/data/viewslug/mnist-original/") ======
    """

    train_set = datasets.fetch_mldata('mnist original')
    print('\n', train_set.data.shape, '\n')

    dimension="""Choose to use the whole dataset dimension (70000) or a smaller part (1000):
1 - Whole dataset dimension
0 - Smaller part \n"""
    choice = int(input(dimension))
    if choice == 0:
        data_dimension = 9000
    elif choice == 1:
        data_dimension = 70000

    # Set X, y
    X, y = train_set.data[:data_dimension], train_set.target[:data_dimension]

    # Plot
    manual_plot(X, y, title="MNIST Original")


def loadDigits():
    """
    ====== Load Digits dataset ("http://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_digits.html") ======
    """

    train_set = datasets.load_digits()
    print('\n', train_set.data.shape, '\n')

    # Set X, y
    X, y = train_set.data, train_set.target

    # Plot
    manual_plot(X, y, title="Load Digits")


"""============================================= MAIN ============================================="""


if __name__ == "__main__":
    menu = """Choose an option:
1 - 20newsgroups
2 - Reuters-21578
3 - MNIST
4 - Load Digits
0 - Exit Program \n
"""
    choice = -1
    while choice != 0:
        choice = int(input(menu))
        if choice == 1:
            twenty_newsgroups()
            choice = -1
        elif choice == 2:
            reuters()
            choice = -1
        elif choice == 3:
            mnist()
            choice = -1
        elif choice == 4:
            loadDigits()
            choice = -1
        elif choice == 0:
            print("Ending Program \n")
