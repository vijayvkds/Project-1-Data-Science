
# Importing the libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler

from sklearn_pandas import DataFrameMapper
from sklearn.compose import ColumnTransformer

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline

import sklearn.metrics as skmet
import pickle

import warnings
warnings.filterwarnings("ignore")

cancerdata = pd.read_csv(r"C:\Users\hp\Desktop\cancerdata.csv")

# Connecting with Database - MySQL

from sqlalchemy import create_engine

# Creating engine which connect to MySQL
user = 'user1' # user name
pw = 'user1' # password
db = 'cancer_db' # database

engine = create_engine(f"mysql+pymysql://{user}:{pw}@localhost/{db}")

# pushing the data into database 
cancerdata.to_sql('cancer', con = engine, index = False)

# loading data from database
sql = 'select * from cancer'
cancerdf = pd.read_sql_query(sql, con = engine)

# Data Preprocessing & EDA

cancerdf['diagnosis'] = np.where(cancerdf['diagnosis'] == 'B', 'Benign', cancerdf['diagnosis'])
cancerdf['diagnosis'] = np.where(cancerdf['diagnosis'] == 'M', 'Malignant', cancerdf['diagnosis'])

cancerdf.drop(['id'], axis = 1, inplace = True) # Excluding id column
cancerdf.info()   # No missing values observed

cancerdf.describe()

# Seggretating input and output variables 
cancerdf_X = pd.DataFrame(cancerdf.iloc[:, 1:])
cancerdf_y = pd.DataFrame(cancerdf.iloc[:, 0])
cancerdf_X.info()

# All numeric features
numeric_features = cancerdf_X.select_dtypes(exclude = ['object']).columns
numeric_features

# Imputation strategy for numeric columns
num_pipeline = Pipeline([('impute', SimpleImputer(strategy = 'mean'))])

# All categorical features
categorical_features = cancerdf_X.select_dtypes(include = ['object']).columns
categorical_features 

# Encoding categorical to numeric variable
categ_pipeline = Pipeline([('label', DataFrameMapper([(categorical_features, OneHotEncoder(drop = 'first'))]))])


# Using ColumnTransformer to transform the columns of an arraay
# Fit the pipeline
preprocess_pipeline = ColumnTransformer([('categorical', categ_pipeline, categorical_features), 
                                       ('numerical', num_pipeline, numeric_features)])
processed = preprocess_pipeline.fit(cancerdf_X)  
processed


# Save the defined pipeline
import joblib
joblib.dump(processed, 'processed1')

import os 
os.getcwd()

# Transform the original data using the pipeline defined above
cancerclean = pd.DataFrame(processed.transform(cancerdf_X), columns = cancerdf_X.columns)  # Cleaned and processed data for ML Algorithm

cancerclean.info()

# Define scaling pipeline
scale_pipeline = Pipeline([('scale', MinMaxScaler())])

preprocess_pipeline2 = ColumnTransformer([('scale', scale_pipeline, cancerclean.columns)]) 

processed2 = preprocess_pipeline2.fit(cancerclean)
processed2

# Save the Scaling pipeline
joblib.dump(processed2, 'processed2')

import os 
os.getcwd()

# Normalized data frame - numerical part of data
cancerclean_n = pd.DataFrame(processed2.transform(cancerclean), columns = cancerclean.columns)

eda = cancerclean_n.describe()
eda

# Output variable Target stored
Y = np.array(cancerdf_y ['diagnosis']) # 

X_train, X_test, Y_train, Y_test = train_test_split(cancerclean_n, Y, test_size = 0.2, random_state = 0)

X_train.shape
X_test.shape

# Model building
knn = KNeighborsClassifier(n_neighbors = 21)

KNN = knn.fit(X_train, Y_train)  # Train the KNN model

# Evaluate the model with train data
pred_train = knn.predict(X_train)  # Predict on train data

pred_train

# Cross table
pd.crosstab(Y_train, pred_train, rownames = ['Actual'], colnames = ['Predictions']) 

skmet.accuracy_score(Y_train, pred_train) # Accuracy measure

# Predict the class on test data
pred_test = knn.predict(X_test)
pred_test

# Evaluate the model with test data
skmet.accuracy_score(Y_test, pred_test)
pd.crosstab(Y_test, pred_test, rownames = ['Actual'], colnames = ['Predictions']) 


# creating empty list variable 
acc = []

# running KNN algorithm from 3 to 50 nearest neighbours - odd numbers

for i in range(3, 50, 2):
    neigh = KNeighborsClassifier(n_neighbors = i)
    neigh.fit(X_train, Y_train)
    train_acc = np.mean(neigh.predict(X_train) == Y_train)
    test_acc = np.mean(neigh.predict(X_test) == Y_test)
    diff = train_acc - test_acc
    acc.append([diff, train_acc, test_acc])
    
acc
    
# Plotting the data accuracies
plt.plot(np.arange(3, 50, 2), [i[1] for i in acc], "ro-")
plt.plot(np.arange(3, 50, 2), [i[2] for i in acc], "bo-")


# Saving the model
knn = KNeighborsClassifier(n_neighbors = 9)
KNN = knn.fit(X_train, Y_train) 

knn_best = KNN
pickle.dump(knn_best, open('knn.pkl', 'wb')) #wb = write in binary format to the file knn.pkl

import os
os.getcwd()
