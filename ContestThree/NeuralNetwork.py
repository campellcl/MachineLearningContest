"""
NeuralNetwork.py
A neural network implementation for contest three.
"""

__author__ = 'Chris Campell'
__version__ = '6/27/2017'

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score

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

def neural_network(x_train, y_train, x_test, y_test):
    mlp = MLPClassifier()
    print("Multi-Layer Perceptron Estimator Parameter Choices:")
    print(mlp.get_params().keys())
    # Normal MLP Classifier on Validation Set:
    clf = MLPClassifier(
        learning_rate='adaptive', verbose=True,
        solver='lbfgs', activation='tanh',
        random_state=1, max_iter=500
    )
    # GridSearchCV Multi-Layer Perceptron Classifier on Validation Set:
    # mlp_pipe = Pipeline([('clf', MLPClassifier(learning_rate='invscaling'))])
    # mlp_param_range = [i for i in range(1, 11)]
    # mlp_eta_range = [1**-i for i in range(1, 5)]
    # mlp_param_grid = [{'clf__hidden_layer_sizes': mlp_param_range}]
    # mlp_gs = GridSearchCV(estimator=mlp_pipe, scoring='accuracy', cv=10, n_jobs=-1)
    # print("Dim x_train: %s" %(x_train.shape,))
    # print("Dim y_train: %s" %(y_train.shape,))
    clf = clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)
    print("Multi-Layer Perceptron Accuracy: %.1f%%" % (100 * accuracy_score(y_true=y_test, y_pred=y_pred)))
    plot_decision_regions(x_test, y_test, clf, 0.02)
    plt.title("MLP Decision Boundaries (Validation Set)")
    plt.axis('equal')
    plt.legend()
    plt.xlabel('x_1')
    plt.ylabel('x_2')
    plt.show()
    # Dump output
    y_eval = clf.predict(x_eval)
    df_out = pd.DataFrame(data=y_eval, index=df_test.index, columns=['Prediction'])
    df_out.to_csv('kaggle_03_submission_01.csv')

if __name__ == '__main__':
    # Read training data:
    df_train = pd.read_csv('kaggle_03_train.csv', index_col='Id')
    # Read testing data:
    df_test = pd.read_csv('kaggle_03_test.csv', index_col='Id')
    # Partition into train_x and train_y:
    x = df_train[['x_1', 'x_2']].values
    y = df_train['y'].values
    x_eval = df_test[['x_1', 'x_2']].values
    # Plot the data for visualization:
    index = y > 0
    plt.scatter(x[index, 0], x[index, 1], label='positive')
    plt.scatter(x[~index, 0], x[~index, 1], label='negative')
    plt.legend()
    plt.axis('equal')
    plt.xlabel('x_1')
    plt.ylabel('x_2')
    plt.title('Initial X and Y Scatter')
    plt.show()
    # Partition into training and testing data:
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=1)
    print("Dim x_train: %s" %(x_train.shape,))
    print("Dim y_train: %s" %(y_train.shape,))
    # y_train = np.reshape(y_train, (y_train.shape[0], 1))
    # print("Dim y_train (reshape): %s" % (y_train.shape,))
    # Create a Multi-Layer Perceptron:
    neural_network(x_train, y_train, x_test, y_test)

