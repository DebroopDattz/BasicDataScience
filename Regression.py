# -*- coding: utf-8 -*-
"""
Created on Mon Aug  4 15:11:34 2025

@author: Debroop
"""
import pandas as pd
import numpy as np
import matplotlib . pyplot as plt
from sklearn . linear_model import LinearRegression
from sklearn . metrics import mean_squared_error , r2_score
# Step 1: Load the dataset
# You can replace this with : pd. read_csv (’ filename .csv ’) if reading from file
data = pd. DataFrame ({'Hours': [1, 2, 3, 4.5 , 5, 6, 7, 8.5 , 9, 10] , 'Marks': [35 , 40, 50, 60, 62, 70, 75, 85, 88, 95]})
# Step 2: Separate the independent variable (X) and dependent variable (y)
X = data['Hours'] # Feature (2D array required by sklearn )
y = data['Marks'] # Target
# Step 3: Create and train the model
model = LinearRegression () # Create a linear regression model
model.fit(X, y) # Fit model to data
# Step 4: Make predictions
y_pred = model . predict (X) # Predict using the trained model
# Step 5: Print model coefficients
print (f" Intercept : { model . intercept_ :.2f}")
print (f" Slope : { model . coef_ [0]:.2 f}")
# Step 6: Evaluate model
mse = mean_squared_error (y, y_pred )
r2 = r2_score (y, y_pred )
print (f" Mean Squared Error : {mse :.2f}")
print(f"$\\text{{R}}^{{2}}$ Score : {r2:.2f}")  # How well the regression line fits the data
# Step 7: Visualization
plt. figure ( figsize =(8 , 5))
plt. scatter (X, y, color ='blue', label ='Actual Marks') # Actual data points
plt. plot (X, y_pred , color ='red', label ='Regression Line') # Regression line
plt. title ('Study Hours vs Marks')
plt. xlabel ('Hours Studied')
plt. ylabel ('Marks Scored')
plt. legend ()
plt. grid ( True )
plt. show ()
# Step 8: Predict on full data and a test case
test_hours = [[6.5]]
predicted_marks = model.predict ( test_hours )
print (f" Predicted marks for 6.5 study hours : { predicted_marks [0]:.2 f}")
