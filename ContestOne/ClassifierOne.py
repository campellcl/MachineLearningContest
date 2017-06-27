from enum import Enum, unique
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

"""
ClassifierOne.py
Performs a variety of visualization methods as performed on the training and test data from competition one.
"""
__author__ = "Chris Campell"
__version__ = "6/26/2017"

def plot_decision_regions(X, y, classifier, resolution=0.02):
    """
    plot_decision_Regions -Plots the decision regions for a classifier along with the sample data.
    :param X: The sample data to be plotted.
    :param y: The target labels of the sample data to be plotted.
    :param classifier: An instance of a classifier object which will be performing the classification.
    :param resolution: TODO: what is this?
    :return:
    """
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
    if debug is Verbosity.verbose:
        print("Beginning Classification Task With Debug Verbose:")
        print("Beginning KNN (n_neighbors=10, dist=Euclidean):")
    else:
        print("Beginning Classification Task With Debug Disabled:")
    # Train KNN:
    knn = KNeighborsClassifier(n_neighbors=10, p=2, metric='minkowski')
    knn.fit(x_train, y_train)
    y_hat = knn.predict(x_test)
    if debug is not Verbosity.no_clf_plots or Verbosity.silent:
        plot_decision_regions(x_test, y_test, classifier=knn)
        plt.title('KNN Decision Regions')
        plt.xlabel('x_1')
        plt.ylabel('x_2')
        plt.legend()
        plt.show()
    if debug is not Verbosity.silent:
        acc_score = accuracy_score(y_true=y_test, y_pred=y_hat)
        print("KNN Accuracy: %.1f%%" %(100 * acc_score))
    # GridSearch KNN on validation set:
    knn_pipe = Pipeline([('clf', KNeighborsClassifier(p=2, metric='minkowski'))])
    param_range = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    knn_param_grid = [{'clf__n_neighbors': param_range}]
    if debug is Verbosity.verbose:
        print("KNN Possible Estimator Parameters:")
        print(knn_pipe.get_params().keys())
    knn_gs = GridSearchCV(estimator=knn_pipe, param_grid=knn_param_grid, scoring='accuracy', cv=10)
    knn_gs = knn_gs.fit(x_train, y_train)
    print("GridSearch KNN Accuracy (Validation Set): %.1f%%" %(100 * knn_gs.best_score_))
    print("GridSearch KNN Best Params (Validation Set):")
    print(knn_gs.best_params_)
    plot_decision_regions(x_test, y_test, classifier=knn_gs)
    plt.title('KNN GridSearchCV Decision Regions (Validation Set)')
    plt.xlabel('x_1')
    plt.ylabel('x_2')
    plt.legend()
    plt.show()
    # GridSearch KNN on withheld testing data:
    knn_gs = GridSearchCV(estimator=knn_pipe, param_grid=knn_param_grid, scoring='accuracy', cv=10)
    knn_gs = knn_gs.fit(x, y)
    y_eval = knn_gs.predict(x_eval)
    print("GridSearch KNN Accuracy (Test Set): %.1f%%" %(100 * knn_gs.best_score_))
    print("GridSearch KNN Best Params (Test Set):")
    print(knn_gs.best_params_)
    # Write prediction to file for upload to kaggle:
    df_out = pd.DataFrame(data=y_eval, index=df_test.index, columns=['Prediction'])
    df_out.to_csv('kaggle_01_submission_01.csv')

if __name__ == '__main__':
    # Specify global unique debug configurations for use with application:
    @unique
    class Verbosity(Enum):
        silent = 1
        verbose = 2
        clf_accuracy_only = 3
        no_clf_plots = 4
    # Specify list of classifiers to perform:
    @unique
    class Classifiers(Enum):
        knn = 1
        perceptron = 2
        adaline = 3

    # Set global runtime flags:
    debug = Verbosity.verbose
    clfs = [Classifiers.knn, Classifiers.perceptron]

    # Read training data:
    df_train = pd.read_csv('kaggle_01_train.csv', index_col='Id')
    # Read testing data:
    df_test = pd.read_csv('kaggle_01_test.csv', index_col='Id')
    if debug is Verbosity.verbose:
        print("Training Data Dimensionality: %s" % (df_train.shape,))
        print("Testing Data Dimensionality: %s" % (df_test.shape,))
    # Partition into x and y:
    x = df_train[['x_1', 'x_2']].values
    y = df_train['y'].values
    x_eval = df_test[['x_1', 'x_2']].values
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
    if debug is Verbosity.verbose:
        print("Training Data 'x_train' Subset Dimensionality: %s" % (x_train.shape,))
        print("Training Data 'y_train' Subset Dimensionality: %s" % (y_train.shape,))
        print("Testing Data 'x_test' Subset Dimensionality: %s" % (x_test.shape,))
        print("Testing Data 'y_test' Subset Dimensionality: %s" % (y_test.shape,))
    main(debug=debug)
