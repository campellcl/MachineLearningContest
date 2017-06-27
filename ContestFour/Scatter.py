"""
Scatter.py
Displays data prior to model construction.
"""

import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

__author__ = "Chris Campell"
__version__ = "6/27/2017"

if __name__ == '__main__':
    # Read training data:
    df_train = pd.read_csv('kaggle_04_train.csv', index_col='Id')
    # Read testing data:
    df_test = pd.read_csv('kaggle_04_test.csv', index_col='Id')
    # Partition into train_x and train_y:
    x = df_train[['x_1', 'x_2']].values
    y = df_train['y'].values
    # Plot the data for visualization:
    index = y > 0
    plt.scatter(x[index, 0], x[index, 1], label='positive')
    plt.scatter(x[~index, 0], x[~index, 1], label='negative')
    plt.legend()
    plt.axis('equal')
    plt.xlabel('x_1')
    plt.ylabel('x_2')
    plt.show()
    # Partition into training and testing data:
    # x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=1)
