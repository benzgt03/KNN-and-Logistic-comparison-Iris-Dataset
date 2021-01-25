import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix,classification_report
import pandas as pd  # import libary ต่างๆที่จะใช้
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from sklearn.datasets import load_iris

#Data Preparation

iris = load_iris()
df_iris = pd.DataFrame(iris.data)
df_iris.columns = iris.feature_names
df_iris['Class'] = iris.target
print(iris.DESCR)
print(df_iris.head()) # Show data
print(df_iris.tail())
print(df_iris.describe()) #Describe all value
print('check number =',df_iris.nunique()) # number of unique variable
print('check null =',df_iris.isnull().sum()) # check Null

#plot comparison

figure1 = plt.figure()
sns.countplot(df_iris['Class'])
sns.pairplot(df_iris, hue = "Class", height=3, markers=["o", "s", "D"])

#train and test model

X_train, X_test,y_train,y_test = train_test_split(iris.data,iris.target, test_size=0.2,random_state= 20)

#DecisionTree

model = DecisionTreeClassifier() #gini
model.fit(X_train,y_train)
y_pred = model.predict(X_test)

#Model Evaluation

print("Model Accuracy : ",model.score(X_test,y_test))
print("Model Accuracy by acc",accuracy_score(y_test,y_pred))
print(classification_report(y_test,y_pred))
print(pd.crosstab(y_test, y_pred, rownames=['True'], colnames=['Predicted'], margins=True))
figure2 = plt.figure()
axis=confusion_matrix(y_test,y_pred)
sns.heatmap(axis,annot=True,cmap='Reds')
plt.title('Confusion Matrix Decision tree')

#Check overfit and underfit

print("Test :",model.score(X_test,y_test))
print("Train :" , model.score(X_train,y_train))
print('It seem like there are same value , the two value are quite comparable so it could not be overfitting .')

#plot tree for Decision Tree

figure3 = plt.figure()
tree.plot_tree(model.fit(X_train, y_train))
plt.title("Decision Tree")

#Train and Test (RandomForest Tree)

X_train, X_test,y_train,y_test = train_test_split(iris.data,iris.target, test_size=0.2,random_state= 35)
rf = RandomForestClassifier()
rf.fit(X_train,y_train)

#Find Feature Importances

feature_important = rf.feature_importances_
feature_important = np.array_split(feature_important,4)
feature_important = np.array(feature_important)
feature_important = np.squeeze(feature_important)
feature_name = df_iris.columns.drop('Class')
print("Feature important = ",feature_important)
figure4 = plt.figure()
barlist = plt.bar(feature_name,feature_important )
barlist[0].set_color('r')
barlist[1].set_color('y')
barlist[2].set_color('g')
plt.ylabel('Corr')
plt.xlabel('Feature')
plt.title('Feature Important')

#Model Evaluation

y_pred = rf.predict(X_test)
print("Model Score RF = ",rf.score(X_test,y_test))
print("Model Accuracy by acc",accuracy_score(y_test,y_pred))
print(classification_report(y_test,y_pred))
print(pd.crosstab(y_test, y_pred, rownames=['True'], colnames=['Predicted'], margins=True))
figure3 = plt.figure()
axis=confusion_matrix(y_test,y_pred)
sns.heatmap(axis,annot=True,cmap='Greens')
plt.title('Confusion Matrix RandomForest Tree')

#Show
plt.show()