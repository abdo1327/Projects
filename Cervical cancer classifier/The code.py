#Import the necessary libraries and functions 
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Normalizer
from sklearn.pipeline import make_pipeline
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix
from scipy.stats import chi2
import random
#Mcnemar function to test the null and alternative hypothesis
def mcnemar(x, y):
    n1 = np.sum(x < y)
    n2 = np.sum(x > y)
    stat = (np.abs(n1-n2)-1)**2 / (n1+n2)
    df = 1
    pval = chi2.sf(stat,1)
    return stat, pval

#read the dataset 
data1 = pd.read_csv("dataframe.csv")
data=data1.drop(columns=['Biopsy'])
target=data1.Biopsy
#split the data to test and train sets 
X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.4, random_state=42)
#Initialize the normalizer and scaler 
scaler = StandardScaler()
norm=Normalizer()

classifiers1 =[]
log1=LogisticRegression()
classifiers1.append(log1)

Knn1= KNeighborsClassifier(n_neighbors=5)
classifiers1.append(Knn1)

tree1 = DecisionTreeClassifier(max_depth=2)
classifiers1.append(tree1)

forest1= RandomForestClassifier(max_depth=2, random_state=0)
classifiers1.append(forest1)

clf1=SVC()
classifiers1.append(clf1)

NB1 = GaussianNB()
classifiers1.append(NB1)

#test the data with Different   classifiers
for clf in classifiers1:
    clf.fit(X_train, y_train)
    y_pred= clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print("The accuracy  is %s"%(acc))
    cm = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix is \n %s "%(cm))
    print("\n\n\n the calssfiers with ")
#test the data with Different   classifiers and normalizer
for clf2 in classifiers1:
    pipeline = make_pipeline( scaler,clf2)
    clf2.fit(X_train, y_train)
    y_pred= clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print("Accuracy is %s"%(acc))
    cm = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix is \n %s "%(cm))
#test the data with Different   classifiers and scaler
for clf3 in classifiers1:
    pipeline = make_pipeline( norm,clf2)
    clf2.fit(X_train, y_train)
    y_pred= clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print("Accuracy is %s"%(acc))
    cm = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix is \n %s "%(cm))
    
# hyperparameter tuning for different classier to find the best parameter for each one
log=LogisticRegression()
solvers = ['newton-cg', 'lbfgs', 'liblinear']
penalty = ['l2']
c_values = [100, 10, 1.0, 0.1, 0.01]
# define grid search
grid = dict(solver=solvers,penalty=penalty,C=c_values)
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
grid_search = GridSearchCV(estimator=log, param_grid=grid, n_jobs=-1, cv=cv, scoring='accuracy',error_score=0)
grid_result = grid_search.fit(X_train,y_train)
# summarize results
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))

KNN = KNeighborsClassifier()
n_neighbors = range(1, 21, 2)
weights = ['uniform', 'distance']
metric = ['euclidean', 'manhattan', 'minkowski']
# define grid search
grid = dict(n_neighbors=n_neighbors,weights=weights,metric=metric)
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
grid_search = GridSearchCV(estimator=KNN, param_grid=grid, n_jobs=-1, cv=cv, scoring='accuracy',error_score=0)
grid_result = grid_search.fit(X_train,y_train)
# summarize results
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))

tree = DecisionTreeClassifier()
criterion = ["gini", "entropy"]
splitter=["best", "random"]
max_depth=[2,3,4,5,6,7,8,9]
# define grid search
grid = dict(criterion=criterion,splitter=splitter,max_depth=max_depth)
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
grid_search = GridSearchCV(estimator=tree, param_grid=grid, n_jobs=-1, cv=cv, scoring='accuracy',error_score=0)
grid_result = grid_search.fit(X_train,y_train)
# summarize results
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))

forest = RandomForestClassifier()
n_estimators = [10, 100, 1000]
max_features = ['sqrt', 'log2']
# define grid search
grid = dict(n_estimators=n_estimators,max_features=max_features)
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
grid_search = GridSearchCV(estimator=forest, param_grid=grid, n_jobs=-1, cv=cv, scoring='accuracy',error_score=0)
grid_result = grid_search.fit(X_train,y_train)
# summarize results
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
SVC = SVC()
kernel = ['poly', 'rbf', 'sigmoid']
C = [50, 10, 1.0, 0.1, 0.01]
gamma = ['scale']
# define grid search
grid = dict(kernel=kernel,C=C,gamma=gamma)
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
grid_search = GridSearchCV(estimator=SVC, param_grid=grid, n_jobs=-1, cv=cv, scoring='accuracy',error_score=0)
grid_result = grid_search.fit(X_train,y_train)
# summarize results
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))


#test the data with Different  classifiers whit the best parameters for each one 
classifiers =[]
log=LogisticRegression(C=.1,solver='newton-cg')
classifiers.append(log)

Knn= KNeighborsClassifier(n_neighbors=7,metric='euclidean',weights = 'uniform')
classifiers.append(Knn)

tree = DecisionTreeClassifier(max_depth=4,criterion='gini',splitter='best')
classifiers.append(tree)

forest= RandomForestClassifier(max_depth=2, random_state=0 ,max_features= 'log2', n_estimators=10)
classifiers.append(forest)


NB = GaussianNB()
classifiers.append(NB)

print("\n\n\n the calssfiers with ")

#print the accuracy, confusion matrix  and accuracy score  for each classifier 
for clf in classifiers:
    clf.fit(X_train, y_train)
    y_pred= clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print("The accuracy is %s "%(acc))
    cm = confusion_matrix(y_test, y_pred)
    print("\nConfusion Matrix \n %s "%(cm))
    print(classification_report(y_test, clf.predict(X_test)))



