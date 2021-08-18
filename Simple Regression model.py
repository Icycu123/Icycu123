from sklearn import linear_model
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
'''Train and Split Test'''
df = pd.read_csv(r"C:\Users\User\Desktop\Python Database\FuelConsumption.csv", encoding='utf-8')
print(df.columns)  # Using index to show categories
cdf = df[['ENGINESIZE', 'CYLINDERS', 'FUELCONSUMPTION_COMB', 'CO2EMISSIONS']]  # Using three categories
msk = np.random.rand(len(df)) < 0.8  # To split data portion for graph
train = cdf[msk]  # 80% of the sample for train
test = cdf[~msk]  # 20% of the sample for test, ~symbol Bitwise not
regr = linear_model.LinearRegression()
train_x = np.asanyarray(train[['ENGINESIZE']])
train_y = np.asanyarray(train[['CO2EMISSIONS']])
regr.fit(train_x, train_y)
print('Coefficients: ', regr.coef_)
print('Intercept: ', regr.intercept_)
# Draw scatter and line
plt.scatter(train.ENGINESIZE, train.CO2EMISSIONS, color='blue')
plt.plot(train_x, regr.coef_[0][0] * train_x + regr.intercept_[0], '-r')
plt.xlabel("Engine size")
plt.ylabel("Emission")
plt.show()
# 20% Test Data to Check Model Error
test_x = np.asanyarray(test[['ENGINESIZE']])  # Create array for target test element
test_y = np.asanyarray(test[['CO2EMISSIONS']])  # Create array for target test element
test_y_ = regr.predict(test_x)  # Predict value
print("Mean absolute error: %.2f" % np.mean(np.absolute(test_y_ - test_y)))  # Model error test,MAE
print("Residual sum of squares (MSE): %.2f" % np.mean((test_y_ - test_y) ** 2))  # Model error test, MSE
print("R2-score: %.2f" % r2_score(test_y, test_y_))  # R square , the higher the score (1), more fit to model
