from enum import Enum, unique
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

"""
ClassifierOne.py
Performs a variety of visualization methods as performed on the training and test data from competition one.
"""
__author__ = "Chris Campell"
__version__ = "6/26/2017"

def plot_decision_regions(X, y, classifier, resolution=0.02):
    # setup marker generator and color map
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])
    # plot the decision surface:
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution), np.arange(x2_min, x2_max, resolution))
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha=0.4, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())
    # plot class samples:
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0], y=X[y == cl, 1], alpha=0.8, c=cmap(idx), marker=markers[idx], label=cl)

def main(debug):
    """
    main -Main method for objects of type DataVisualization. TODO: Finish method header.
    :param debug: A flag indicating to what degree debug information should be printed.
    :return:
    """
    if debug:
        print("Beginning Classification Task With Debug Verbose:")
        print("Beginning KNN:")
    else:
        print("Beginning Classification Task With Debug Disabled:")
    # Train KNN:
    knn = KNeighborsClassifier(n_neighbors=10, p=2, metric='minkowski')
    knn.fit(x_train, y_train)
    plot_decision_regions(x_train, y_train, classifier=knn)
    plt.xlabel('x_1')
    plt.ylabel('x_2')
    plt.legend()
    plt.show()

if __name__ == '__main__':
    # Specify global unique debug configurations for use with application:
    @unique
    class Verbosity(Enum):
        verbose_debug = 1
        debug = 2
        silent = 3
    # Set global debug flag:
    debug = Verbosity.verbose_debug

    # Read training data:
    df_train = pd.read_csv('kaggle_01_train.csv', index_col='Id')
    # Read testing data:
    test = pd.read_csv('kaggle_01_test.csv', index_col='Id')
    if debug is Verbosity.verbose_debug:
        print("Training Data Dimensionality: %s" % (df_train.shape,))
        print("Testing Data Dimensionality: %s" % (test.shape,))
    # Partition into x and y:
    x = df_train[['x_1', 'x_2']].values
    y = df_train['y'].values
    # Visualize the Data:
    index = y > 0
    plt.scatter(x[index, 0], x[index, 1], label='positive')
    plt.scatter(x[~index, 0], x[~index, 1], label='negative')
    plt.legend()
    plt.axis('equal')
    plt.xlabel('x_1')
    plt.ylabel('x_2')
    plt.show()
    # Partition Data into Train, Test Split:
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=1)
    if debug is Verbosity.verbose_debug:
        print("Training Data 'x_train' Subset Dimensionality: %s" % (x_train.shape,))
        print("Training Data 'y_train' Subset Dimensionality: %s" % (y_train.shape,))
        print("Testing Data 'x_test' Subset Dimensionality: %s" % (x_test.shape,))
        print("Testing Data 'y_test' Subset Dimensionality: %s" % (y_test.shape,))
    main(debug=debug)
