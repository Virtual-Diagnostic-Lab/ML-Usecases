#importing packages
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier


iris_df=pd.read_csv('iris_csv.csv')
print(iris_df.head(10))
print(iris_df.info())
print(iris_df.groupby('class').size())
# print(iris_df.isnull.values.any())
plt.figure(figsize=(15,10))
plt.subplot(2,2,1)
sns.boxplot(x='class',y='sepallength',data=iris_df)
plt.subplot(2,2,2)
sns.boxplot(x='class',y='sepalwidth',data=iris_df)
plt.subplot(2,2,3)
sns.boxplot(x='class',y='petallength',data=iris_df)
plt.subplot(2,2,4)
sns.boxplot(x='class',y='petalwidth',data=iris_df)

#splitting data
X=iris_df.values[:,0:4]
y=iris_df.values[:,4]
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=0)

#ml model
#SVC
svc_classifier=SVC(max_iter=1000 ,gamma='scale')
svc_classifier.fit(X_train,y_train)
svc_y_pred=svc_classifier.predict(X_test)
svc_accuracy=metrics.accuracy_score(svc_y_pred,y_test)
print('SVM_accuracy:',round(svc_accuracy,2)*100)

#Decision Tree 
dectree_classifier=DecisionTreeClassifier()
dectree_classifier.fit(X_train,y_train)
dectree_y_pred=dectree_classifier.predict(X_test)
dectree_accuracy=metrics.accuracy_score(y_test,dectree_y_pred)
print('decision_tree_accuracy:',round(dectree_accuracy,2)*100)



