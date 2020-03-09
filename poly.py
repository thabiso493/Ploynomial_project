import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

x_train = [[13],[15],[23],[29],[30],[35]] # Number of crates in a warehouse
y_train = [[10],[12],[13.5],[14.7],[15],[15.6]] # Percentage increase in value of crates

x_test = [[12],[13],[21],[29],[30],[35]] # Number of crates in a warehouse
y_test = [[9],[11],[13.5],[14.7],[15],[15.6]] # Percentage increase in value of crates

# Plotting a prediction and training the linear regression
regressor = LinearRegression()
regressor.fit(x_train,y_train)
xx = np.linspace(0,36,150)
yy = regressor.predict(xx.reshape(xx.shape[0], 1))
plt.plot(xx,yy)

# Setting the degree of the Polynomial Regression model
model_degree_featurizer = PolynomialFeatures(degree=2)

# Transforming an input data matrix into a new matrix of a given degree
x_train_quad = model_degree_featurizer.fit_transform(x_train)
y_train_quad = model_degree_featurizer.transform(x_test)

# Training and testing the Regression model
reg_quad = LinearRegression()
reg_quad.fit(x_train_quad,y_train)
xx_quad = model_degree_featurizer.transform(xx.reshape(xx.shape[0],1))

# Plotting the graph
plt.plot(xx, reg_quad.predict(xx_quad),c='r',linestyle='--')
plt.title('Percentage value increase of crates of stock as per crate stored')
plt.xlabel('Number of crates of stock in warehouse')
plt.ylabel('Percentage increase in value of crates')
plt.axis([0,40,0,20])
plt.grid(True)
plt.scatter(x_train,y_train)
plt.show()
print(x_train)
print(x_train_quad)
print(y_train)
print(y_train_quad)
