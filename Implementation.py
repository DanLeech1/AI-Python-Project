#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Imports
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression



#DATA PREPERATION

# Used to flip rows and columns of the dataset
#pd.read_csv('medicalexpenditure.csv', header=None).T.to_csv('medicalexpenditure2.csv', header=False, index=False)

# Open CSV File
df = pd.read_csv("medicalexpenditure2.csv")

# Print first few rows
print("head output:\n" , df.head(), "\n==================\n")

# Print type of each attribute
print("info output\n")
df.info()
print("\n==================\n")

# Print Basic Statistics
print("describe output:\n" , df.describe(), "\n==================\n")

# Print Duplicates
print("duplicate output:\n" , df.duplicated().sum(), "\n==================\n")

# Make new df and remove duplicates
newdf = df.copy(deep=True)
newdf=newdf.drop_duplicates()

# Print Nulls
print("is null output: \n" , df.isnull().sum(), "\n==================\n")

# Drop null columns
newdf.drop('* Medical expenses includes of inpatients, outpatients and dental treatments.', inplace=True, axis=1) 
newdf.drop('**approximately, 1 JPY to 0.01$', inplace=True, axis=1) 

# Drop null rows
newdf.dropna(inplace=True)



#IMPLEMENTATION

# Factors relevant to Medical Expenses
x= newdf[['Medical expenses for inpatients (JPY)',
          'Medical expenses for outpatients (JPY)',
          'Number of doctors', 'Number of beds',
          'Number of nurses', 'Income (10,000JPY)']]

# Medical Expenses
y= newdf['Medical expenses* (JPY**)']

# 80/20 split for training/testing
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

# Using LinearRegression to train the data
lin_reg = LinearRegression() 
lin_reg.fit(x_train, y_train)



#RESULTS

# Return the coefficient of determination
print ("Sklearn coefficient of determination for multivariate regression is:", 
lin_reg.score(x_test, y_test)) 

# Plot accuracy of prediction
plt.figure()
plt.scatter(y_train, lin_reg.predict(x_train),color = 'b', label='Predicted values')
plt.scatter(y_test, lin_reg.predict(x_test) ,color = 'r', label='Test values')
plt.legend()








