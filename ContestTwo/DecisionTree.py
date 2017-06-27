"""
DecisionTree.py
A decision tree classifier for the second contest.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV

__author__ = 'Chris Campell'
__version__ = '6/27/2017'

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


def decision_tree(x_train, x_test, y_train, y_test):
    clf = DecisionTreeClassifier()
    print("DecisionTreeClassifier Estimator Parameter Choices:")
    print(clf.get_params().keys())
    # GridSearch Decision Tree on Validation Set:
    tree_pipe = Pipeline([('clf', DecisionTreeClassifier(criterion='gini', random_state=1))])
    param_range = [i for i in range(1,11)]
    tree_param_grid = [{'clf__max_depth': param_range}]
    tree_gs = GridSearchCV(estimator=tree_pipe, param_grid=tree_param_grid, scoring='accuracy', cv=10)
    tree_gs = tree_gs.fit(x_train, y_train)
    print("GridSearchCV Decision Tree Accuracy: %.1f%%" %(100 * tree_gs.best_score_))
    print("GridSearchCV Decision Tree Parameters:")
    print(tree_gs.best_params_)
    plot_decision_regions(x_test, y_test, classifier=tree_gs)
    plt.title("Decision Tree GridSearchCV Decision Boundaries (Validation Set)")
    plt.axis('equal')
    plt.xlabel('x_1')
    plt.ylabel('x_2')
    plt.legend()
    plt.show()
    # Record predictions for kaggle upload:
    y_eval = tree_gs.predict(x_eval)
    df_out = pd.DataFrame(data=y_eval, index=df_test.index, columns=['Prediction'])
    df_out.to_csv('kaggle_02_submission_01.csv')

if __name__ == '__main__':
    # Read training data:
    df_train = pd.read_csv('kaggle_02_train.csv', index_col='Id')
    # Read testing data:
    df_test = pd.read_csv('kaggle_02_test.csv', index_col='Id')
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
    # Create a decision tree:
    decision_tree(x_train, x_test, y_train, y_test)
