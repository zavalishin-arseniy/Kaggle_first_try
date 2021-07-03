import numpy as np
import scipy
from scipy import linalg, optimize
import pandas as pd
import xlrd
import matplotlib.pyplot as plt
import seaborn
import sklearn
from sklearn import datasets
from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet, LogisticRegression
from sklearn import metrics
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans, DBSCAN
import scipy
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram

#//////////////////////////////////////////////////////////////////////////////
#Линейная регрессия
# boston = datasets.load_boston()
# print(boston.keys())
# print(boston.DESCR[100:1300])
# boston_df=pd.DataFrame(boston.data, columns=boston.feature_names)
# print(boston_df.head(5))

# lr = LinearRegression()
# model = lr.fit(boston.data, boston.target)
# fw_df=pd.DataFrame(list(zip(boston.feature_names, model.coef_)))
# fw_df.columns=['f','w']
# # print(fw_df)
# # print(model.intercept_)
# pred=model.predict(boston.data)
# print(pred[:10])
# tr_df=pd.DataFrame(list(zip(pred, boston.target)))
# fw_df.columns=['p','t']
# print(tr_df.head(5))
#//////////////////////////////////////////////////////////////////////////////
#Логистическая регрессия
# cancer=datasets.load_breast_cancer()
# log_r=LogisticRegression()
# class_log=log_r.fit(cancer.data, cancer.target)
# pred=class_log.predict(cancer.data)
# prob=class_log.predict_proba(cancer.data)
# print(prob[:10])
# print('Acc:{}'.format(class_log.score(cancer.data,cancer.target)))
# print(class_log.get_params())
#
# print('Acc: {:.2f}'.format(metrics.accuracy_score(cancer.target,pred)))
# print('AUC: {:.2f}'.format(metrics.roc_auc_score(cancer.target,pred)))
# print('F1: {:.2f}'.format(metrics.f1_score(cancer.target,pred)))


#//////////////////////////////////////////////////////////////////////////////
#Разделение датасета
# heart=pd.read_excel(r"C:\Users\User\Documents\ML\heart.xls", "heart")
# cancer=datasets.load_breast_cancer()
# log_r=LogisticRegression()
# xtrain, xtest, ytrain, ytest = train_test_split(cancer.data, cancer.target, 0.2, random_state=12)

#//////////////////////////////////////////////////////////////////////////////
#Разные линейные модели
# cancer = datasets.load_breast_cancer()
#
# lr = LinearRegression()
# lasso = Lasso
# ridge = Ridge()
# elnet = ElasticNet()
# for model in [lasso, ridge, elnet]:
#     x_train, x_test, y_train, y_test = train_test_split(cancer.data, cancer.target, test_size=0.2)
#
#     model.fit(x_train, y_train)
#     pred=model.predict(cancer.data)
#     print(model.__class__)

# //////////////////////////////////////////////////////////////////////////////
#Кросс-валидация
# iris = datasets.load_iris()
# lr=LogisticRegression()
# cv=KFold(n_splits=5)
# for split_id, (train_id, test_id) in enumerate(cv.split(iris.data)):
#     x_train, x_test = iris.data[train_id], iris.data[test_id]
#     y_train, y_test = iris.target[train_id], iris.target[test_id]
#
#     lr.fit(x_train,y_train)
#     score=lr.score(x_test,y_test)
#     print('split {} score: {:2f}'.format(split_id,score))
#
# cv_score = cross_val_score(lr, iris.data, iris.target,
#                            scoring='accuracy', cv=cv)
# print(cv_score)
# print(cv_score.mean())


#//////////////////////////////////////////////////////////////////////////////
#Случайный лес и градинетный бустинг
# iris = datasets.load_iris()
#
# rf =RandomForestClassifier(n_estimators=100, random_state=42)
# xtrain, xtest, ytrain, ytest = train_test_split(iris.data, iris.target,test_size=0.3,
#                                                 stratify=iris.target, random_state=42)
# rf_model=rf.fit(xtrain,ytrain)
# #важность параметров
# fi = list(zip(iris.feature_names, rf_model.feature_importances_))
# fi_df=pd.DataFrame(fi, columns=['f', 'i'])
# print(fi_df)
# тоже самое для градинетного бустинга GradientBoosrtingClass/Reg
#да-да тоже самое можно и с регрессией, а не с классификацией делать
#можно посмотреть параметры, это может помочь
#обрати особое внимание на oob_score у леса, это проверка у деревьев на остальных элементах


#//////////////////////////////////////////////////////////////////////////////
#Немного линала

# x= np.array([[1,59],[1, 1]])
# #выводит количество строк и столбцов
# print(x.shape)
# #выводит массив, с элементами, меньше заданного
# print(x[x<2])
# #разварачивает двумерный массив в одномерной
# print(x.flatten())
#
# y= np.array([[3,5],[7.2]])

# x+y = add(x,y) / x-y =subset(x,y)

#произведение multiply(x,y) поэлементное умножение
#произведние dot(x,y) скалярное

#//////////////////////////////////////////////////////////////////////////////


# heart=pd.read_excel(r"C:\Users\User\Documents\ML\heart.xls", "heart")
# sat=pd.read_csv(r"C:\Users\User\Documents\ML\o2Saturation.csv", "o2Saturation")
#
# print(heart.head(5))
# print(sat.head(5))
#

#//////////////////////////////////////////////////////////////////////////////
#Рисуем графики
# plt.hist(heart['sex'])
# plt.show()
#fig, axes = plt.subplots(nrows=,ncols=,figsize=) fig это наша фигура, а axes это несколько графиков в ней
#axes[][].plot(...) рисуем график того что в скобках
#axes[][].set_titel меняем название
#for axes, row_axes in enumirate(axes):
#   for column, ax in enumirate(row_axes): для прохода по холстам
# fig.tight_layout чтобы все помещалось
#kde
#//////////////////////////////////////////////////////////////////////////////
# Немного пандас
# print(heart.head(5))
# print(heart.columns)
# print(heart[['age', 'sex', 'cp', 'trtbps', 'chol', 'fbs', 'restecg', 'thalachh',
#        'exng', 'oldpeak', 'slp', 'caa', 'thall', 'output']].head(5))
# print(heart.iloc[-1,4])
# print(heart.describe())
#heart.groupby(['group name']).groups
# merge для совмещения таблиц

#usetype={'sub_1':1, 'sub_2':2,...}
#heart['sub_name'].map(usetype)
#heart.applay(lamda x: f(x))

#//////////////////////////////////////////////////////////////////////////////
#K-means

# X,y=make_blobs(n_samples=150, n_features=2, centers=4, random_state=1)
#
# k=KMeans(n_clusters=2, init='k-means++', n_init=10, random_state=1)
# k.fit(X)
# l=k.labels_
# print(l)
#
# c=[]
# for i in range(2,8):
#     nk = KMeans(n_clusters=i, random_state=1)
#     nk.fit(X)
#
#     c.append(nk.inertia_)
#
# plt.plot(range(2,8),c)
# plt.show()

#//////////////////////////////////////////////////////////////////////////////

#Иерархическая класстеризация x=df.iloc[:, :1].values
#x=(x-x.mean(axes=0))/x.std(axes=0)
#привести данные к одинаковому виду
# z=linkage(x, method='average',metric='euclidean')
# dend=dendrogram(Z)
# lab = fcluster(Z, 2.2, criterion='distance')

#//////////////////////////////////////////////////////////////////////////////

#DBSCAN
# db = DBSCAN(eps=, min_samples=, metric='haversine', algorithm='ball_tree')
# db.fit(X)
