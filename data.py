# -*- coding: utf-8 -*-
"""
Created on Fri Apr 26 16:31:20 2019

@author: ELİF NUR
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn import preprocessing


def loadData(fromPath, LabelColumnName, labelCount):  # 此方法读取csv文件并更改标签特性

    data_ = pd.read_csv(fromPath)
    if labelCount == 2:  # 如果数据集包含有多重类型的攻击，则把攻击数据当作Anormal，常规数据看作Normal；这个数据集只能识别是否发生攻击，不能识别攻击类别
        dataset = data_
        dataset[LabelColumnName] = dataset[LabelColumnName].apply(
            {'DoS': 'Anormal', 'BENIGN': 'Normal', 'DDoS': 'Anormal', 'PortScan': 'Anormal'}.get)
    else:
        dataset = data_
    data = dataset[LabelColumnName].value_counts()  # 查看数据集的label有多少种不同的值，及每个label有多少个数据
    data.plot(kind='pie')  # 用饼状图表示数据分布
    featureList = dataset.drop([LabelColumnName], axis=1).columns  # 删除label这一列，创建特征值列表
    return dataset, featureList  # 返回带有label的数据集和特征值列表


def datasetSplit(df, LabelColumnName):  # 该方法将数据集分离为X和y.
    labelencoder = LabelEncoder()  # 编码label
    df.iloc[:, -1] = labelencoder.fit_transform(df.iloc[:, -1])  # 在最前面一列设置为label的先拟合数据，然后转化它将其转化为标准形式
    X = df.drop([LabelColumnName], axis=1)  # 设置X值为不含有label的数据集
    X = np.array(X)  # X转换格式为矩阵格式
    X = X.T  # 将矩阵X转置，即行列互换
    for column in X:  # Control of values in X
        median = np.nanmedian(column)  # 计算每列中位数
        column[np.isnan(column)] = median  # 将该列缺失值位置设置为中位数
        column[column == np.inf] = 0  # 正无穷的位置设置为0
        column[column == -np.inf] = 0  # 负无穷的位置设置为0
    X = X.T  # 将矩阵转置复原
    scaler = preprocessing.MinMaxScaler()
    X = scaler.fit_transform(X)  # 将X归一化，将属性缩放到一个指定的最大和最小值（通常是1-0）之间
    y = df[[LabelColumnName]]  # 设置Y为标签值，未标准化的label
    return X, y


def train_test_dataset(df, LabelColumnName):  # 该方法将数据集分离为X_train、X_test、y_train和y_test.
    labelencoder = LabelEncoder()
    df.iloc[:, -1] = labelencoder.fit_transform(df.iloc[:, -1])
    X = df.drop([LabelColumnName], axis=1)
    y = df[[LabelColumnName]]
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, test_size=0.3, random_state=0, stratify=y)  # 切分数据集，random_state就是为了保证程序每次运行都分割一样的训练集合测试集，依据标签y，按原数据y中各类比例，分配给train和test，使得train和test中各类数据的比例与原数据集一样
    X_train = np.array(X_train)  # 矩阵X
    X_train = X_train.T  # X转置
    for column in X_train:
        median = np.nanmedian(column)
        column[np.isnan(column)] = median
        column[column == np.inf] = 0
        column[column == -np.inf] = 0
    X_train = X_train.T
    y_train = np.array(y_train)
    y_train = y_train.T
    for column in y_train:
        median = np.nanmedian(column)
        column[np.isnan(column)] = median
        column[column == np.inf] = 0
        column[column == -np.inf] = 0
    y_train = y_train.T
    X_test = np.array(X_test)
    X_test = X_test.T
    for column in X_test:
        median = np.nanmedian(column)
        column[np.isnan(column)] = median
        column[column == np.inf] = 0
        column[column == -np.inf] = 0
    X_test = X_test.T
    y_test = np.array(y_test)
    y_test = y_test.T
    for column in y_test:
        median = np.nanmedian(column)
        column[np.isnan(column)] = median
        column[column == np.inf] = 0
        column[column == -np.inf] = 0
    y_test = y_test.T

    return X_train, X_test, y_train, y_test

