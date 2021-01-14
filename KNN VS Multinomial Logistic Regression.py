import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, confusion_matrix,recall_score,f1_score,classification_report
import pandas as pd  # import libary ต่างๆที่จะใช้
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
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

#Train and Scailing
X_train, X_test,y_train,y_test = train_test_split(iris.data,iris.target, test_size=0.5,random_state= 44)
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
#preparation for loop
Knn = np.arange(1,9)
train_sc = np.empty(len(Knn))
test_sc = np.empty(len(Knn))

#find K that best fit for model
for i,k in enumerate(Knn) :
    model = KNeighborsClassifier(n_neighbors= k)
    model.fit(X_train,y_train)
    test_sc[i] = model.score(X_test,y_test)
    max_score = max(test_sc)
    max_score_index = Knn[i]
    print("Score =",max_score,"K =",max_score_index)
figure2 = plt.figure()
plt.plot(Knn , test_sc)
#Use K to make the best model

Knn1 = KNeighborsClassifier(n_neighbors=3) # find best k from loop
Knn1.fit(X_train,y_train)

#Model Evaluation for KNN
print("Model Accuracy : ",Knn1.score(X_test,y_test))
y_pred = Knn1.predict(X_test)
print(classification_report(y_test,y_pred))
figure3 = plt.figure()
print(pd.crosstab(y_test, y_pred, rownames=['True'], colnames=['Predicted'], margins=True))
axis=confusion_matrix(y_test,y_pred)
sns.heatmap(axis,annot=True,cmap='Reds')
plt.title('Confusion Matrix KNN')

# Train and Scaling
X_train, X_test,y_train,y_test = train_test_split(iris.data,iris.target, test_size=0.5,random_state= 44)
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Multiclass Logistic Regression
lg = LogisticRegression(multi_class='multinomial', solver='lbfgs')
lg.fit(X_train,y_train)
y_pred = lg.predict(X_test)

# Model Evaluation for Logistic Regression
print("Model Score : ",accuracy_score(y_test,y_pred))
print(classification_report(y_test,y_pred))
print(pd.crosstab(y_test, y_pred, rownames=['True'], colnames=['Predicted'], margins=True))
figure4 = plt.figure()
axis=confusion_matrix(y_test,y_pred)
sns.heatmap(axis,annot=True,cmap='Greens')
plt.title('Confusion Matrix Logistic')

#Conclusion

figure6 = plt.figure()
list_cc = [max_score , accuracy_score(y_test,y_pred) ]
list_cc_name = ['KNN', 'Logistic Regression']
barlist = plt.bar(list_cc_name,list_cc )
barlist[0].set_color('r')
plt.ylabel('Precision')
plt.xlabel('Model')
plt.title('Conclusion Comparison')

#Show plot

plt.show()