from tpot import TPOTClassifier
from sklearn.model_selection import train_test_split
import pandas as pd 
import numpy as np

titanic  = pd.read_csv("https://raw.githubusercontent.com/insaid2018/Term-1/master/Data/Casestudy/titanic_train.csv")     # Importing training dataset using pd.read_csv
titanic.rename(columns={'Survived': 'class'}, inplace=True)
titanic['Sex'] = titanic['Sex'].map({'male':0,'female':1})
titanic['Embarked'] = titanic['Embarked'].map({'S':0,'C':1,'Q':2})
titanic = titanic.fillna(-999)
pd.isnull(titanic).any()
from sklearn.preprocessing import MultiLabelBinarizer
mlb = MultiLabelBinarizer()
CabinTrans = mlb.fit_transform([{str(val)} for val in titanic['Cabin'].values])
titanic_new = titanic.drop(['Name','Ticket','Cabin','class'], axis=1)
assert (len(titanic['Cabin'].unique()) == len(mlb.classes_)), "Not Equal" #check correct encoding done
titanic_new = np.hstack((titanic_new.values,CabinTrans))
titanic_class = titanic['class'].values

training_indices, validation_indices = training_indices, testing_indices = train_test_split(titanic.index, stratify = titanic_class, train_size=0.75, test_size=0.25)
tpot = TPOTClassifier(verbosity=3, max_time_mins=2, max_eval_time_mins=0.04, generations=200, population_size=200, n_jobs=-1)
tpot.fit(titanic_new[training_indices], titanic_class[training_indices])

