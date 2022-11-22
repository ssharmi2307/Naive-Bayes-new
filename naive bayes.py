# -*- coding: utf-8 -*-
"""
Created on Tue Nov  1 12:31:40 2022

@author: Gopinath
"""
#Loading dataset
import pandas as pd
import numpy as np
train=pd.read_csv("SalaryData_Train.csv")
test=pd.read_csv("SalaryData_Test.csv")
list(train)
train.shape
train.head()
train.describe()
list(test)
test.shape
test.head()
test.describe()

#data cleaning
train[train.duplicated()].shape
train[train.duplicated()]
train=train.drop_duplicates()
train.isnull().sum()

test[test.duplicated()].shape
test[test.duplicated()]
test=test.drop_duplicates()
test.isnull().sum()

pd.crosstab(train['occupation'], train['Salary'])
pd.crosstab(train['workclass'], train['Salary'])
pd.crosstab(train['occupation'], train['workclass'])

#data visulaization
import seaborn as sns
import matplotlib.pyplot as plt
#for train dataset
sns.countplot(x='Salary',data= train)
plt.xlabel('Salary')
plt.ylabel('count')
plt.show()
train['Salary'].value_counts()

#for test dataset
sns.countplot(x='Salary',data= test)
plt.xlabel('Salary')
plt.ylabel('count')
plt.show()
test['Salary'].value_counts()


sns.scatterplot(train['occupation'],train['workclass'],hue=train['Salary'])
pd.crosstab(train['Salary'], train['occupation']).mean().plot(kind='bar')
pd.crosstab(train['Salary'], train['workclass']).mean().plot(kind='bar')
pd.crosstab(train['Salary'], train['education']).mean().plot(kind='bar')



##Preprocessing the data. As, there are categorical variables
columns = ["workclass","education","maritalstatus","occupation","relationship","race","sex","native"]
from sklearn.preprocessing import LabelEncoder
number = LabelEncoder()
for i in columns:
        train[i]= number.fit_transform(train[i])
        test[i]=number.fit_transform(test[i])
train
test

#train and test
x_train = train[train.columns[0:13]].values
y_train = train[train.columns[13]].values
x_test = test[test.columns[0:13]].values
y_test = test[test.columns[13]].values

##Normalmization
def norm_func(i):
    x = (i-i.min())/(i.max()-i.min())
    return (x)
x_train
x_test
y_train
y_test


#applying naive bayes for classification
from sklearn.naive_bayes import MultinomialNB as MB
M_model=MB()
train_multi=M_model.fit(x_train,y_train).predict(x_train)
test_multi=M_model.fit(x_train,y_train).predict(x_test)
train_acc_multi=np.mean(train_multi==y_train)
train_acc_multi

test_acc_multi=np.mean(test_multi==y_test)
test_acc_multi
# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix,accuracy_score
confusion_matrix = confusion_matrix(y_test, test_multi)
#print the matrix
confusion_matrix
print(accuracy_score(y_test,test_multi))#accuracy score

## GaussianNB for numerical data
from sklearn.naive_bayes import GaussianNB as GB
G_model=GB()
train_gau=G_model.fit(x_train,y_train).predict(x_train)
test_gau=G_model.fit(x_train,y_train).predict(x_test)
train_acc_gau=np.mean(train_gau==y_train)
train_acc_gau

test_acc_gau=np.mean(test_gau==y_test)
test_acc_gau

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
confusion_matrix = confusion_matrix(y_test, test_gau)
#print the matrix
confusion_matrix
#calculating the accuracy of this model w.r.t. this dataset
from sklearn.metrics import accuracy_score
print(accuracy_score(y_test,test_multi))
















