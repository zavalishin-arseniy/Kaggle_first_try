import numpy as np
from tqdm import tqdm
import scipy
from scipy import linalg, optimize
import pandas as pd
import seaborn as sns
import xlrd
import matplotlib.pyplot as plt
import seaborn
import math
import sklearn
from sklearn import datasets
from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet, LogisticRegression
from sklearn import metrics
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans, DBSCAN
import scipy
from scipy import stats
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram

# ss=pd.read_csv(r"sample_submission.csv")

def split_array(array):
    test = []
    train = np.random.choice(array, size= int(0.75*len(array)), replace=False)
    for i in range(len(array)):
        if not(array[i] in train):
            test.append(array[i])
    return train, test

train=pd.read_csv(r"train.csv")
test=pd.read_csv(r"test.csv")

# for i in train.columns:
#     for j in train.columns:
#         if i != j and scipy.stats.spearmanr(train[i], train[j])[0] > 0.6:
#             print(scipy.stats.spearmanr(i, j)[0])


# if train.isnull().values.any():
#     print('yes')
# else:
#     print('no')

# corr = train.corr()
# co = sns.heatmap(corr)
# plt.show()


# print(ss)
# print('/////////////////////////////')
# print(train)
# print('/////////////////////////////')
# print(test)


# train_tagret = train.target.values
# train_data = train.iloc[:, :51]
# print(train_data)
# print('class 1 : {:.2f}\nclass 2 : {:.2f}\nclass 3 : {:.2f}\nclass 4 : {:.2f}'.format(np.sum(train_tagret=='Class_1')/float(len(train_tagret)),
#                                                                                       np.sum(train_tagret=='Class_2')/float(len(train_tagret)),
#                                                                                       np.sum(train_tagret=='Class_3')/float(len(train_tagret)),
#                                                                                       np.sum(train_tagret=='Class_4')/float(len(train_tagret))))

for i in range(len(train['target'])):
    train['target'][i]=int(train['target'][i][-1])

w_class1 = np.where(train['target'] == 1)[0]
w_class2 = np.where(train['target'] == 2)[0]
w_class3 = np.where(train['target'] == 3)[0]
w_class4 = np.where(train['target'] == 4)[0]

len_1 = len(w_class1)
len_2 = len(w_class2)
len_3 = len(w_class3)
len_4 = len(w_class4)

min_class = min(len_1, len_2, len_3, len_4)

w_class1_downsampled = np.random.choice(w_class1, size=min(4*len_1, min_class), replace=False)
w_class2_downsampled = np.random.choice(w_class2, size=min(4*len_2, min_class), replace=False)
w_class3_downsampled = np.random.choice(w_class3, size=min(4*len_3, min_class), replace=False)
w_class4_downsampled = np.random.choice(w_class4, size=min(4*len_4, min_class), replace=False)


#попытка в кросс-валидацию
max = 0
# model_fix
for i in tqdm(range(100)):
    train_array = []
    test_array = []

    train_part, test_part = split_array(w_class1_downsampled)
    train_array.append(train_part)
    test_array.append(test_part)
    train_part, test_part = split_array(w_class2_downsampled)
    train_array.append(train_part)
    test_array.append(test_part)
    train_part, test_part = split_array(w_class3_downsampled)
    train_array.append(train_part)
    test_array.append(test_part)
    train_part, test_part = split_array(w_class4_downsampled)
    train_array.append(train_part)
    test_array.append(test_part)

    index_train = [a for b in train_array for a in b]
    index_test = [a for b in test_array for a in b]

    index_train.sort()

    ind = train ['id']. isin (index_train)
    new_train_data = train[ind]
    train_target = new_train_data.target.values.astype('int')
    train_data = new_train_data.iloc[:, :51].astype('int')

    ind = train ['id']. isin (index_test)
    new_test_data = train[ind]
    test_target = new_train_data.target.values.astype('int')
    test_data = new_train_data.iloc[:, :51].astype('int')

    rf = RandomForestClassifier(n_estimators=200, random_state=30, max_depth=20)
    model = rf.fit(train_data, train_target)
    model_pred = model.predict(test_data)
    if max < metrics.f1_score(test_target,model_pred, average='weighted'):
        max = metrics.f1_score(test_target,model_pred, average='weighted')
        model_fix = model
    print(metrics.f1_score(test_target,model_pred, average=None))
    print(metrics.f1_score(test_target,model_pred, average='macro'))
    print(metrics.f1_score(test_target,model_pred, average='micro'))
    print(metrics.f1_score(test_target,model_pred, average='weighted'))

# Действующая классификация
rf_predictions = model_fix.predict(test)
# # Вероятности для каждого класса
rf_probs = model_fix.predict_proba(test)
# print(rf_probs)


# index_arr = []
# index_arr.append(list(w_class1_downsampled))
# index_arr.append(list(w_class2_downsampled))
# index_arr.append(list(w_class3_downsampled))
# index_arr.append(list(w_class4_downsampled))
# index_arr = [a for b in index_arr for a in b]
# index_arr.sort()
# ind = train ['id']. isin (index_arr)
# new_data = train[ind]
# # print(new_data)
#
# train_target = new_data.target.values.astype('int')
# train_data = new_data.iloc[:, :51].astype('int')
#
#
# rf = RandomForestClassifier(n_estimators=200, random_state=30, max_depth=20)
#
# model = rf.fit(train_data, train_target)


# Действующая классификация
# rf_predictions = fix_model.predict(test)
# # Вероятности для каждого класса
# rf_probs = fix_model.predict_proba(test)
# print(rf_probs)

result_data = pd.DataFrame(rf_probs, columns=['Class_1', 'Class_2', 'Class_3', 'Class_4'])
result_data.index = result_data.index+100000
print(result_data)
result_data.to_csv('output.csv', index=True)
# model_pred = model.predict(train_data)
# print(metrics.f1_score(train_target,model_pred, average=None))
# print(metrics.f1_score(train_target,model_pred, average='macro'))
# print(metrics.f1_score(train_target,model_pred, average='micro'))
# print(metrics.f1_score(train_target,model_pred, average='weighted'))

