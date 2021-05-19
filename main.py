'''
Aakash Das

Data Science And Business Analytics Internship(May 2021)

TASK 1 - Prediction using Supervised ML
'''
# libraries used
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split  # used in line 32
from sklearn.linear_model import LinearRegression #used in line 35

from sklearn import metrics #used in line 74 and onwards

# importing data
data = pd.read_csv("http://bit.ly/w-data")
df = data  # creating dataframe
print(df)

# Plotting scores
plt.scatter(df['Hours'], df['Scores'], color='blue')
plt.title('Hours vs Percentage Plot')
plt.xlabel('Hours Studied')
plt.ylabel("Percent Scored")
plt.show()

# Prepping data for training
x = df.iloc[:, :-1].values
y = df.iloc[:, 1].values

# Split the Data in two
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.25, random_state = 0)

#Training 
model = LinearRegression()
model.fit(x_train, y_train)

#plotting linear regression
line = model.coef_*x + model.intercept_
plt.title('Regression Graph',size=14)
plt.scatter(x, y, color = 'red')
plt.plot(x, line)
plt.show()

#making Predictions
print(x_test)
y_pred = model.predict(x_test)

#comparing actual vs predicted marks
compare = pd.DataFrame({'Actual Marks': y_test, 'Predicted Marks': y_pred})
print(compare)

#Compare graphically the Predicted Marks with the Actual Marks
plt.scatter(x=x_test, y=y_test, color='red')
plt.plot(x_test, y_pred, color='Blue')
plt.title('Actual vs Predicted', size=14)
plt.ylabel('Percentage', size=11)
plt.xlabel('Study Hours', size=11)
plt.show()


#Checking the accuracy of training and test scores
print('Test Score')
print(model.score(x_test, y_test))
print('Training Score')
print(model.score(x_train, y_train))

#Testing with custom input

hrs = [7.85]
answer = model.predict([hrs])
print(" Respective Score = {}".format(round(answer[0],3)))

#metrics 
print('Mean Absolute Error(MAE) :', metrics.mean_absolute_error(y_test, y_pred))
print('Mean Squared Error(MSE) :', metrics.mean_squared_error(y_test, y_pred))
print('Root Mean Squared Error(RMSE) :', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
