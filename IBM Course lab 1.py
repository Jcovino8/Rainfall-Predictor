#IBM Course lab 1

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

url= "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-ML0101EN-SkillsNetwork/labs/Module%202/data/FuelConsumptionCo2.csv"

df=pd.read_csv(url)
df.describe()

cdf = df[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_COMB','CO2EMISSIONS']]
cdf.sample(9)

viz = cdf[['CYLINDERS','ENGINESIZE','FUELCONSUMPTION_COMB','CO2EMISSIONS']]
viz.hist()
plt.show()

plt.scatter(cdf.FUELCONSUMPTION_COMB, cdf.CO2EMISSIONS,  color='blue')
plt.xlabel("FUELCONSUMPTION_COMB")
plt.ylabel("Emission")
plt.show()

plt.scatter(cdf.ENGINESIZE, cdf.CO2EMISSIONS,  color='blue')
plt.xlabel("Engine size")
plt.ylabel("Emission")
plt.xlim(0,27)
plt.show()

#my code below
# Instructions: Plot __CYLINDER__ against CO2 Emission, to see how linear their relationship is.

plt.scatter(cdf.CYLINDERS, cdf.CO2EMISSIONS,  color='blue')
plt.ylabel("CO2 Emission")
plt.xlabel("CYLINDERS")
plt.show()

# Given 
X = cdf.ENGINESIZE.to_numpy()
y = cdf.CO2EMISSIONS.to_numpy()

# Model training starts below
from sklearn.model_selection import train_test_split
#This line splits the data into 4 groups - 80% used for training, 20% for testing
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)

#The outputs are one-dimensional NumPy arrays or vectors
type(X_train), np.shape(X_train), np.shape(X_train)

#Build a simple linear regression model

from sklearn import linear_model

# create a model object
regressor = linear_model.LinearRegression()

# train the model on the training data
# X_train is a 1-D array but sklearn models expect a 2D array as input for the training data, with shape (n_observations, n_features).
# So we need to reshape it. We can let it infer the number of observations using '-1'.
regressor.fit(X_train.reshape(-1, 1), y_train)

# Print the coefficients
print ('Coefficients: ', regressor.coef_[0]) # with simple linear regression there is only one coefficient, here we extract it from the 1 by 1 array.
print ('Intercept: ',regressor.intercept_)

#Visualize the model output
plt.scatter(X_train, y_train,  color='blue')
plt.plot(X_train, regressor.coef_ * X_train + regressor.intercept_, '-r')
plt.xlabel("Engine size")
plt.ylabel("Emission")


#Model evaluation
from sklearn.metrics import mean_absolute_error, mean_squared_error, root_mean_squared_error, r2_score

# Use the predict method to make test predictions
y_test_ = regressor.predict( X_test.reshape(-1,1))

print("Mean absolute error: %.2f" % mean_absolute_error(y_test_, y_test))
print("Mean squared error: %.2f" % mean_squared_error(y_test_, y_test))
print("Root mean squared error: %.2f" % root_mean_squared_error(y_test_, y_test))
print("R2-score: %.2f" % r2_score( y_test_, y_test) )

# Practice exercise: Plot the regression model result over the test data instead of the training data. Visually evaluate whether the result is good.
plt.scatter(X_test, y_test,  color='blue')
plt.plot(X_test, regressor.coef_ * X_test + regressor.intercept_, '-r')
plt.xlabel("Engine size")
plt.ylabel("Emission")

#Practice exercise: Select the fuel consumption feature from the dataframe and split the data 80%/20% into training and testing set
X = cdf.FUELCONSUMPTION_COMB.to_numpy()
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)

#Practice exercise: Train a linear regression model using the data you created
regr = linear_model.LinearRegression()
regr.fit(X_train.reshape(-1, 1), y_train)

#Practice exercise: Use the model to make test predictions on the fuel consumption testing data.
y_test_ = regr.predict(X_test.reshape(-1,1))

#Practice exercise: Calculate and print the Mean Squared Error of the test predictions.
print("Mean squared error: %.2f" % mean_squared_error(y_test_, y_test))


