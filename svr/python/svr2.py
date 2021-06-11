from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:2].values.astype(float)
y = dataset.iloc[:, 2:3].values.astype(float)
sc_X = StandardScaler()
sc_y = StandardScaler()
X = sc_X.fit_transform(X)
y = sc_y.fit_transform(y)
regressor = SVR(kernel='rbf')
regressor.fit(X, y)
y_pred = sc_y.inverse_transform((regressor.predict(sc_X.transform(np.array([[6.5]])))))
plt.scatter(X, y, color='magenta')
plt.plot(X, regressor.predict(X), color='green')
plt.title('Truth or Bluff (Support Vector Regression Model)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()
