#!/usr/bin/env python
# coding: utf-8


import numpy as np
import pandas as pd

path = "../dataset/titanic/"
dt_train = pd.read_csv(path+"train.csv")

# Now, we will create a new column saying if a person is married or not
## NOTE: In this column doesn't matter if a person is married and spouse isn't on the boat, 
## that people will be considered married
married = np.zeros((len(dt_train['Name']), 1)).astype(int)
for n in dt_train['Name']:
    count = 0
    pos = dt_train[dt_train['Name']==n].index.values.astype(int)[0]
    if "(" in n:
        married[pos] = 1
    else:
        for t in dt_train['Name']:
            if n in t:
                count += 1
        if count > 1:
            married[pos] = 1

# Saving on the data the new column
dt_train['Married'] = married

# Now, we'll create a new column consider if the spouse of a person is on the boat.
spouses_on_boat = np.zeros((len(dt_train['Name']),1)).astype(int)
for i in range(len(dt_train['SibSp'])):
    s = dt_train.iloc[i]
    if int(s['Married']) and s['SibSp'] > 0:
        spouses_on_boat[i] = 1
dt_train['Spouse_on_boat'] = spouses_on_boat # Saving the new column

# Creating a column that will say if a person has sisters on the boat
sisters = np.zeros((len(dt_train['Name']), 1)).astype(int)
for i in range(len(dt_train['SibSp'])):
    s = dt_train.iloc[i]['SibSp'] - dt_train.iloc[i]['Spouse_on_boat']
    sisters[i] = s
dt_train['Sisters'] = sisters # Saving the new column

# How "Cabin" has only 204 registers we will remove from data
dt_train = dt_train.drop(columns=['Cabin']) #another way: dt_train.drop('Cabin', axis=1, inplace=True)

# How we have 714 of 891 registers in "age" column, we will fill in the column with mean of all ages
dt_train['Age'].fillna(round(dt_train['Age'].mean(), 2), inplace=True)

# Removing the off data in 'Embarked' column.
dt_train = dt_train.dropna(subset=['Embarked'])

# Changing male to 0 and female to 1
dt_train = dt_train.replace(['male', 'female'], [0, 1])
dt_train['Sex'].unique()

# removing discrepant values from 'Fare'
dt_train = dt_train[dt_train['Fare'] <= 263]

from sklearn.preprocessing import LabelEncoder
#Creating a sparse matrix for categorical column 'Embarked'
le = LabelEncoder()
dt_train['Embarked'] = le.fit_transform(dt_train['Embarked'])

# removing text columns from data
dt_train.drop('Name', axis=1, inplace=True)
dt_train.drop('Ticket', axis=1, inplace=True)
dt_train.head(6)

from sklearn.preprocessing import StandardScaler
# Scaling the data to use in ml algoritm
scaler = StandardScaler()
target = dt_train["Survived"]
dt_train.drop("PassengerId", axis=1, inplace=True)
#dt_train.drop("Survived", axis=1, inplace=True) # take off target column from data
dt_train_scaled = scaler.fit_transform(dt_train)
dt_train_scaled = pd.DataFrame(dt_train_scaled)
dt_train_scaled

dt_train_scaled.drop(0, axis=1, inplace=True)

from sklearn.linear_model import LogisticRegression
#aplying the Logistic Regression algoritm
log_r = LogisticRegression()
log_r.fit(dt_train_scaled, target)

proba = log_r.predict_proba(dt_train_scaled)

proba_fig = plt.figure()
p = proba_fig.add_subplot()
p.hist(proba)

### Only thing that we need now is test!!
### tests
### tests