print(__doc__)

from keys import MY_KEY, BOT_TOKEN

import os
import numpy as np
import matplotlib.pyplot as plt
import telepot
import tarfile
import os.path
import itertools
import re

from sklearn import datasets
from sklearn.datasets import get_data_home
from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.externals.six.moves import html_parser
from sklearn.externals.six.moves import urllib
from datetime import datetime
from glob import glob


bot = telepot.Bot(BOT_TOKEN)


def _not_in_sphinx():
    # Hack to detect whether we are running by the sphinx builder
    return '__file__' in globals()

class ReutersParser(html_parser.HTMLParser):
    """Utility class to parse a SGML file and yield documents one at a time."""

    def __init__(self, encoding='latin-1'):
        html_parser.HTMLParser.__init__(self)
        self._reset()
        self.encoding = encoding

    def handle_starttag(self, tag, attrs):
        method = 'start_' + tag
        getattr(self, method, lambda x: None)(attrs)

    def handle_endtag(self, tag):
        method = 'end_' + tag
        getattr(self, method, lambda: None)()

    def _reset(self):
        self.in_title = 0
        self.in_body = 0
        self.in_topics = 0
        self.in_topic_d = 0
        self.title = ""
        self.body = ""
        self.topics = []
        self.topic_d = ""

    def parse(self, fd):
        self.docs = []
        for chunk in fd:
            self.feed(chunk.decode(self.encoding))
            for doc in self.docs:
                yield doc
            self.docs = []
        self.close()

    def handle_data(self, data):
        if self.in_body:
            self.body += data
        elif self.in_title:
            self.title += data
        elif self.in_topic_d:
            self.topic_d += data

    def start_reuters(self, attributes):
        self.reuters = attributes

    def end_reuters(self):
        self.body = re.sub(r'\s+', r' ', self.body)
        self.docs.append({'title': self.title,
                          'body': self.body,
                          'topics': self.topics,
                          'reuters': self.reuters})
        self._reset()

    def start_title(self, attributes):
        self.in_title = 1

    def end_title(self):
        self.in_title = 0

    def start_body(self, attributes):
        self.in_body = 1

    def end_body(self):
        self.in_body = 0

    def start_topics(self, attributes):
        self.in_topics = 1

    def end_topics(self):
        self.in_topics = 0

    def start_d(self, attributes):
        self.in_topic_d = 1

    def end_d(self):
        self.in_topic_d = 0
        self.topics.append(self.topic_d)
        self.topic_d = ""


def stream_reuters_documents(data_path=None):
    """Iterate over documents of the Reuters dataset.

    The Reuters archive will automatically be downloaded and uncompressed if
    the `data_path` directory does not exist.

    Documents are represented as dictionaries with 'body' (str),
    'title' (str), 'topics' (list(str)) keys.

    """

    DOWNLOAD_URL = ('http://archive.ics.uci.edu/ml/machine-learning-databases/'
                    'reuters21578-mld/reuters21578.tar.gz')
    ARCHIVE_FILENAME = 'reuters21578.tar.gz'

    if data_path is None:
        data_path = os.path.join(get_data_home(), "reuters")
    if not os.path.exists(data_path):
        """Download the dataset."""
        print("downloading dataset (once and for all) into %s" %
              data_path)
        os.mkdir(data_path)

        def progress(blocknum, bs, size):
            total_sz_mb = '%.2f MB' % (size / 1e6)
            current_sz_mb = '%.2f MB' % ((blocknum * bs) / 1e6)
            if _not_in_sphinx():
                print('\rdownloaded %s / %s' % (current_sz_mb, total_sz_mb),
                      end='')

        archive_path = os.path.join(data_path, ARCHIVE_FILENAME)
        urllib.request.urlretrieve(DOWNLOAD_URL, filename=archive_path,
                                   reporthook=progress)
        if _not_in_sphinx():
            print('\r', end='')
        print("untarring Reuters dataset...")
        tarfile.open(archive_path, 'r:gz').extractall(data_path)
        print("done.")

    parser = ReutersParser()
    for filename in glob(os.path.join(data_path, "*.sgm")):
        for doc in parser.parse(open(filename, 'rb')):
            yield doc


positive_class = [('earn', 1), ('acq', 2), ('money-fx', 3), ('crude', 4),
    ('grain', 5), ('trade', 6), ('interest', 7), ('ship', 8), ('wheat', 9),
     ('corn', 10)]


def filter_documents(doc_iter, content):
    """Filter through an iterator for desired topics in training set.
     Return a tuple X_text, y.
    """
    data = [(content.format(**doc), check_topics(doc))
          for doc in itertools.islice(doc_iter, 21578)
          if check_topics(doc)]
    if not len(data):
      return np.asarray([], dtype=int), np.asarray([], dtype=int)
    X_text, y = zip(*data)
    return X_text, np.asarray(y, dtype=int)


def check_topics(doc, pos_class = positive_class):
    """Verifies if a document contains at least one of the chosen topics
    """
    for topic in pos_class:
      if topic[0] in doc['topics']:
          return topic[1]
    return False


def manual_plot(X, y, title):
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
        yy_mean = np.array(yy)
        std_dev = np.array(standard_deviation)
        if name == "Naive Bayes":
            plt.plot(xx, yy, '.-', color="r", label=name)
            plt.fill_between(xx, yy_mean-std_dev, yy_mean+std_dev, alpha=0.2, color='r')
            plt.errorbar(xx, yy, yerr=standard_deviation, linestyle='None', color='r')
        elif name == "Decision Tree":
            plt.plot(xx, yy, '.-', color="g", label=name)
            plt.fill_between(xx, yy_mean-std_dev, yy_mean+std_dev, alpha=0.2, color='g')
            plt.errorbar(xx, yy, yerr=standard_deviation, linestyle='None', color='g')
        print("Standard Deviation: {}".format(standard_deviation))

    plt.grid()
    plt.title(title)
    plt.legend(loc="upper right")
    plt.xlabel("Proportion train")
    plt.ylabel("Test Error Rate")
    end = datetime.now() - start
    save_picture(plt, title, end)
    plt.show()


def notify_end_process(directory, execution_time):
    bot.sendMessage(MY_KEY, "PyCharm ha finito in {} secondi.".format(execution_time))
    bot.sendPhoto(MY_KEY, open(directory, 'rb'))


def save_picture(plt, dataset_name, execution_time):
    name_fig = dataset_name

    directory = "/home/federico/PycharmProjects/Progetto_AI/Plotted/" + name_fig

    image_number = 0
    while os.path.exists('{}{:d}.png'.format(directory, image_number)):
        image_number += 1
    plt.savefig('{}{:d}.png'.format(directory, image_number))
    print("Execution time: {}".format(execution_time))
    notify_end_process('{}{:d}.png'.format(directory, image_number), execution_time)


def loadDigits():
    # Load Digits

    train_set = datasets.load_digits()
    print('\n', train_set.data.shape, '\n')

    # Set X, y
    X, y = train_set.data, train_set.target

    manual_plot(X, y, title="Load Digits")


def twenty_newsgroups():
    # convert data for evalutation
    train_set = datasets.fetch_20newsgroups(subset='train', shuffle=True, random_state=42)
    X_train_counts = CountVectorizer().fit_transform(train_set.data)
    X_train_tfidf = TfidfTransformer().fit_transform(X_train_counts)
    print('\n', X_train_tfidf.shape, '\n')

    # Set X, y
    X, y = X_train_tfidf, train_set.target

    # Plot
    manual_plot(X, y, title="20 Newsgroups")


def reuters():
    # Reuters

    # Set X, y
    cont = u'{title}\n\n{body}'
    iterator = stream_reuters_documents()
    X, y = filter_documents(iterator, cont)
    X_train_counts = CountVectorizer().fit_transform(X)
    X_train_tfidf = TfidfTransformer().fit_transform(X_train_counts)
    X = X_train_tfidf
    print('\n', X.shape, '\n')

    # Plot
    manual_plot(X, y, title="Reuters - 21578")


def mnist():
    #MNIST Original

    train_set = datasets.fetch_mldata('mnist original')
    print('\n', train_set.data.shape, '\n')

    # Set X, y
    X, y = train_set.data, train_set.target

    # Plot
    manual_plot(X, y, title="MNIST Original")


def california_housing():
    # California_Housing

    train_set = fetch_california_housing()
    print('\n', train_set.data.shape, '\n')

    # Set X, y
    X, y = MinMaxScaler().fit_transform(train_set.data), (train_set.target*1000).astype(int)

    # Plot
    manual_plot(X, y, title="California Housing")


def leukemia():
    # Leukemia

    train_set = datasets.fetch_mldata('leukemia')
    print('\n', train_set.data.shape, '\n')

    # Set X, y
    X, y = MinMaxScaler().fit_transform(train_set.data), MinMaxScaler().fit_transform(train_set.target)

    # Plot
    manual_plot(X, y, title="Leukemia")


# Pseudo-main
if __name__ == "__main__":
    menu = """Choose an option:
1 - 20newsgroups
2 - Reuters-21578
3 - MNIST
4 - Load Digits
5 - California Housing
6 - Leukemia
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
        elif choice == 5:
            california_housing()
            choice = -1
        elif choice == 6:
            leukemia()
            choice = -1
        elif choice == 0:
            print("Ending Program \n")