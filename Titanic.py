#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import random as rnd

from sklearn.ensemble import RandomForestClassifier

import warnings
warnings.filterwarnings("ignore")


# In[2]:


# Read in data
print("Reading in data")

train_df = pd.read_csv("train.csv")
test_df = pd.read_csv("test.csv")

combined_data = [train_df, test_df]


# In[3]:


# Analyse data by generating graphs for each feature
print ("Generating graphs for each feature")

import seaborn as sns
import matplotlib.pyplot as plt

class_graph = sns.FacetGrid(train_df, aspect=1)
class_graph.map(sns.barplot, "Pclass", "Survived", capsize=.1)
plt.show()

sex_graph = sns.FacetGrid(train_df, aspect=1)
sex_graph.map(sns.barplot, "Sex", "Survived", capsize=.1)
plt.show()

age_graph = sns.FacetGrid(train_df, hue="Survived", aspect=1)
age_graph.map(sns.kdeplot, "Age")
age_graph.set( xlim=( 0 , train_df[ "Age" ].max() ) )
age_graph.add_legend()
plt.show()

sibsp_graph = sns.FacetGrid(train_df, aspect=1)
sibsp_graph.map(sns.barplot, "SibSp", "Survived", capsize=.1)
plt.show()

parch_graph = sns.FacetGrid(train_df, aspect=1)
parch_graph.map(sns.barplot, "Parch", "Survived", capsize=.1)
plt.show()

fare_graph = sns.FacetGrid(train_df, hue="Survived", aspect=1)
fare_graph.map(sns.kdeplot, "Fare")
fare_graph.set( xlim=( 0 , train_df[ "Fare" ].max() ) )
fare_graph.add_legend()
plt.show()

embarked_graph = sns.FacetGrid(train_df, aspect=1)
embarked_graph.map(sns.barplot, "Embarked", "Survived", capsize=.1)
plt.show()


# In[4]:


# Handle title mapping
print("Generating 'Title' feature")

title_mapping = {"Master": 1, "Miss": 2, "Mr": 3, "Mrs": 4, "Special": 5}

for df in combined_data:
    df["Title"] = df.Name.str.extract("([A-Za-z]+)\.", expand=False)

for df in combined_data:
    df["Title"] = df["Title"].replace([
        "Capt", "Col", "Countess", "Don", "Dr", "Jonkheer", "Lady", "Major", "Mlle", "Mme", "Ms", "Rev", "Sir"
    ], "Special")
    
for df in combined_data:
    df["Title"] = df["Title"].map(title_mapping)
    df["Title"] = df["Title"].fillna(0)
    
title_graph = sns.FacetGrid(train_df, aspect=1)
title_graph.map(sns.barplot, "Title", "Survived", capsize=.1)
plt.show()


# In[5]:


# Guess missing ports
print("Guessing missing ports")

for df in combined_data:
    df["Embarked"] = df["Embarked"].fillna(train_df.Embarked.dropna().mode()[0])


# In[6]:


# Guess missing fares
print("Guessing missing fares")

for df in combined_data:
    df["Fare"] = df["Fare"].fillna(train_df.Fare.dropna().mean())


# In[7]:


# Convert sex and port to int
print("Converting columns to ints")

for df in combined_data:
    df["Sex"] = df["Sex"].map( {"female": 0, "male": 1} ).astype(int)
    df["Embarked"] = df["Embarked"].map( { "C": 0, "Q": 1, "S": 2 } ).astype(int)


# In[8]:


# Guess missing ages
print("Guessing missing ages")

predicted_ages = np.zeros((2,3))

for df in combined_data:
    for i in range(0, 2):
        for j in range (0, 3):
            guess_df = df[(df["Sex"] == i) & (df["Pclass"] == j+1)]["Age"].dropna()
            
            age_guess = guess_df.mean()
            
            predicted_ages[i,j] = int(age_guess/0.5 + 0.5) * 0.5
            
    for i in range (0, 2):
        for j in range(0, 3):
            df.loc[ (df.Age.isnull()) & (df.Sex == i) & (df.Pclass == j+1), "Age"] = predicted_ages[i,j]
            
    df["Age"] = df["Age"].astype(int)


# In[9]:


# Categorise ages
print("Categorising ages")

for df in combined_data:    
    df.loc[ df["Age"] <= 15, "Age"] = 0
    df.loc[(df["Age"] > 15) & (df["Age"] <= 30), "Age"] = 1
    df.loc[(df["Age"] > 30) & (df["Age"] <= 60), "Age"] = 2
    df.loc[ df["Age"] > 60, "Age"] = 3

age_graph = sns.FacetGrid(train_df, aspect=1)
age_graph.map(sns.barplot, "Age", "Survived", capsize=.1)
plt.show()


# In[10]:


# Categorise fares
print("Categorising fares")

for df in combined_data:
    df.loc[ df["Fare"] <= 25, "Fare"] = 0
    df.loc[ df["Fare"] > 25, "Fare"] = 1
    df["Fare"] = df["Fare"].astype(int)
    
fare_graph = sns.FacetGrid(train_df, aspect=1)
fare_graph.map(sns.barplot, "Fare", "Survived", capsize=.1)
plt.show()


# In[11]:


# Generate family category
print("Generating 'WithFamily' feature")

for df in combined_data:
    df["WithFamily"] = 0
    df.loc[df["Parch"] > 0, "WithFamily"] = 1
    df.loc[df["SibSp"] > 0, "WithFamily"] = 1
    
family_graph = sns.FacetGrid(train_df, aspect=1)
family_graph.map(sns.barplot, "WithFamily", "Survived", capsize=.1)
plt.show()


# In[12]:


# Drop useless columns
print("Dropping useless columns")

train_df = train_df.drop(["PassengerId", "Name", "Ticket", "Cabin", "Parch", "SibSp"], axis=1)
test_df = test_df.drop(["Name", "Ticket", "Cabin", "Parch", "SibSp"], axis=1)

combined_data = [train_df, test_df]


# In[13]:


# Create training data
print("Creating training data objects")

X_train = train_df.drop("Survived", axis=1)
y_train = train_df["Survived"]
X_test  = test_df.drop("PassengerId", axis=1).copy()


# In[14]:


train_df.head()


# In[15]:


# Train
print("Training Random Forest (without hyper-parameter optimisation)")

clf = RandomForestClassifier(n_estimators=100)
clf.fit(X_train, y_train)
Y_pred = clf.predict(X_test)
clf.score(X_train, y_train)
acc = clf.score(X_train, y_train)
print("Scored: " + str(acc))

# Scored 0.79425 on Kaggle


# In[16]:


# Verify model accuracy with cross-validation
print("Verifying model with cross-validation")

from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score

kf = KFold(n_splits=10)
outcomes = []
split = 0
for train_index, test_index in kf.split(X_train):
    split += 1
    X_train_part, X_test_part = X_train.values[train_index], X_train.values[test_index]
    y_train_part, y_test_part = y_train.values[train_index], y_train.values[test_index]
    predictions = clf.predict(X_test_part)
    accuracy = accuracy_score(y_test_part, predictions)
    outcomes.append(accuracy)
    print("Fold " + str(split) + " accuracy " + str(accuracy))
mean_outcome = np.mean(outcomes)
print("Mean " + str(mean_outcome))


# In[17]:


# Export predictions to CSV file
print("Exporting predictions to CSV")

passenger_id = test_df.PassengerId
test = pd.DataFrame( { "PassengerId": passenger_id , "Survived": Y_pred } )

test.to_csv("titanic_prediction.csv", index = False)


# In[18]:


# Re-train model using GridSearchCV to determine optimal hyper-parameters
print("Training Random Forest (with hyper-paramater optimisation)")
print("May take some time...")

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

clf = RandomForestClassifier()

parameters = {"n_estimators": [4, 6, 9, 15, 50, 100], 
              "max_features": ["log2", "sqrt", "auto"], 
              "criterion": ["entropy", "gini"],
              "max_depth": [2, 3, 5, 10, 15, 40], 
              "min_samples_split": [2, 3, 5, 8, 15],
              "min_samples_leaf": [1, 5, 8, 14, 25]
             }

# Example best params - uncomment this to save yourself 30 minutes:
#parameters = {"n_estimators": [15], 
#              "max_features": ["sqrt"], 
#              "criterion": ["gini"],
#              "max_depth": [40], 
#              "min_samples_split": [15],
#              "min_samples_leaf": [1]
#             }

grid = GridSearchCV(clf, parameters, scoring="accuracy", cv=10, n_jobs=4, verbose=1)
grid = grid.fit(X_train, y_train)

clf = grid.best_estimator_

print("Best hyper-parameters found: " + str(grid.best_params_))


# In[19]:


# Fit the best algorithm to the data. 
print("Fitting best model to training data")

clf.fit(X_train, y_train)

Y_pred = clf.predict(X_test)
clf.score(X_train, y_train)
acc = clf.score(X_train, y_train)
print("Scored: " + str(acc))

# Scored 0.79425 on Kaggle


# In[20]:


# Verify model accuracy with cross-validation
print("Verifying model with cross-validation")

from sklearn.model_selection import KFold

kf = KFold(n_splits=10)
outcomes = []
split = 0
for train_index, test_index in kf.split(X_train):
    split += 1
    X_train_part, X_test_part = X_train.values[train_index], X_train.values[test_index]
    y_train_part, y_test_part = y_train.values[train_index], y_train.values[test_index]
    predictions = clf.predict(X_test_part)
    accuracy = accuracy_score(y_test_part, predictions)
    outcomes.append(accuracy)
    print("Fold " + str(split) + " accuracy " + str(accuracy))
mean_outcome = np.mean(outcomes)
print("Mean " + str(mean_outcome))


# In[21]:


# Export predictions to CSV file
print("Exporting predictions to CSV")

passenger_id = test_df.PassengerId
test = pd.DataFrame( { "PassengerId": passenger_id , "Survived": Y_pred } )

test.to_csv("titanic_prediction_grid.csv", index = False)


# In[22]:


# Generate some new data and predict results to contextualise model
print("Creating custom datapoints to analyse results")

# Guess my chance of survival
# class sex age fare embarked title family
# 2     1   1   0    2        3     1

print(clf.predict_proba([[2, 1, 1, 0, 2, 3, 1]]))

# Guess my father"s chance of survival
# class sex age fare embarked title family
# 2     1   2   0    2        3     1

print(clf.predict_proba([[2, 1, 2, 0, 2, 3, 1]]))

# Guess my mother"s chance of survival
# class sex age fare embarked title family
# 2     0   2   0    2        4     1

print(clf.predict_proba([[2, 0, 2, 0, 2, 4, 1]]))

# Guess my sister"s chance of survival
# class sex age fare embarked title family
# 2     0   1   0    2        2     1

print(clf.predict_proba([[2, 0, 1, 0, 2, 2, 1]]))

# Guess ideal scenario chance of survival
# class sex age fare embarked title family
# 1     0   1   1    0        2     1

print(clf.predict_proba([[1, 0, 0, 1, 0, 2, 1]]))

# Guess worst case scenario chance of survival
# class sex age fare embarked title family
# 3     1   1   0    2        3     0

print(clf.predict_proba([[3, 1, 1, 0, 2, 3, 1]]))


# In[23]:


# Try a different model - Logistic Regression Model
print("Training Logistic Regression (with hyper-paramater optimisation)")

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV

clf = LogisticRegression()

parameters = {"dual": [True, False], 
              "tol": [1e-4, 1e-3, 1e-5, 1e-6],
              "C": [1.0, 10.0, 0.1, 100.0],
              "fit_intercept": [True, False],
              "max_iter": [100, 1000, 10000]
             }

grid = GridSearchCV(clf, parameters, scoring="accuracy", cv=10, n_jobs=4, verbose=1)
grid = grid.fit(X_train, y_train)

clf = grid.best_estimator_

print("Best params: " + str(grid.best_params_))

clf.fit(X_train, y_train)
Y_pred = clf.predict(X_test)
clf.score(X_train, y_train)
acc = round(clf.score(X_train, y_train) * 100, 2)
print("Accuracy: " + str(acc))

passenger_id = test_df.PassengerId
test = pd.DataFrame( { "PassengerId": passenger_id , "Survived": Y_pred } )
test.to_csv("titanic_prediction_lr.csv", index = False)


# In[24]:


# Try a different model - Linear SVM Model

from sklearn.svm import LinearSVC
from sklearn.model_selection import GridSearchCV

# Choose the type of classifier. 
clf = LinearSVC()

# Choose some parameter combinations to try
parameters = {"dual": [True, False],
              "C": [0.1, 1.0, 10.0, 100.0, 1000.0],
              "max_iter": [100, 1000, 10000]
             }

# Run the grid search
grid = GridSearchCV(clf, parameters, scoring="accuracy", cv=10, n_jobs=4, verbose=1)
grid = grid.fit(X_train, y_train)

# Set the clf to the best combination of parameters
clf = grid.best_estimator_

print(grid.best_params_)

clf.fit(X_train, y_train)
Y_pred = clf.predict(X_test)
clf.score(X_train, y_train)
acc = round(clf.score(X_train, y_train) * 100, 2)
print(acc)

passenger_id = test_df.PassengerId
test = pd.DataFrame( { "PassengerId": passenger_id , "Survived": Y_pred } )
test.to_csv("titanic_prediction_svm.csv", index = False)


# In[25]:


# Try a different model - GaussianNB model

from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import GridSearchCV

# Choose the type of classifier. 
clf = GaussianNB()

# Choose some parameter combinations to try
parameters = {}

# Run the grid search
grid = GridSearchCV(clf, parameters, scoring="accuracy", cv=10, n_jobs=4, verbose=1)
grid = grid.fit(X_train, y_train)

# Set the clf to the best combination of parameters
clf = grid.best_estimator_

print(grid.best_params_)

clf.fit(X_train, y_train)
Y_pred = clf.predict(X_test)
clf.score(X_train, y_train)
acc = round(clf.score(X_train, y_train) * 100, 2)
print(acc)

passenger_id = test_df.PassengerId
test = pd.DataFrame( { "PassengerId": passenger_id , "Survived": Y_pred } )
test.to_csv("titanic_prediction_nb.csv", index = False)

