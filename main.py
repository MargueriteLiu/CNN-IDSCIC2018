# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.

from __future__ import print_function
import numpy as np
np.random.seed(1337)  # for reproducibility


import pandas as pd

from sklearn.preprocessing import Normalizer
import numpy as np


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press ⌘F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    traindata = pd.read_csv('/Users/marguerite/Documents/IDS-2018/CSV/Friday-02-03-2018_TrafficForML_CICFlowMeter.csv', header=None)

    X = traindata.iloc[1:4, 3:78]
    Y = traindata.iloc[1:4, 79]


    #print(X)
    #print('###############')
    #print(Y)


    scaler = Normalizer().fit(X)  # 特征值预处理、正则化
    trainX = scaler.transform(X)  # 通过找中心和缩放等实现标准化


    y_train = np.array(Y)
    print(y_train)
    print('###############')
    print(trainX)


# See PyCharm help at https://www.jetbrains.com/help/pycharm/
