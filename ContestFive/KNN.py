"""
KNN.py
K-Nearest Neighbors implementation for contest five.
"""

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

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

if __name__ == '__main__':
    # Read training data:
    df_train = pd.read_csv('kaggle_05_train.csv', index_col='Id')
    # Read testing data:
    df_test = pd.read_csv('kaggle_05_test.csv', index_col='Id')
    # Partition into train_x and train_y:
    x = df_train[['x_1', 'x_2']].values
    y = df_train['y'].values
    x_eval = df_test[['x_1', 'x_2']].values
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=1)
    # GridSearch KNN on validation set:
    knn_pipe = Pipeline([('clf', KNeighborsClassifier(p=2, metric='minkowski'))])
    param_range = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    knn_param_grid = [{'clf__n_neighbors': param_range}]
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
    # Record predictions for kaggle upload:
    y_eval = knn_gs.predict(x_eval)
    df_out = pd.DataFrame(data=y_eval, index=df_test.index, columns=['Prediction'])
    df_out.to_csv('kaggle_05_submission_01.csv')
