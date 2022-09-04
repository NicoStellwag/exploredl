## imports

import torch
import torch.nn as nn
import numpy as np
from sklearn import datasets
import matplotlib.pyplot as plt

##
## prepare data

# get a linear regression problem from sklearn
# shapes will be (n_samples, n_features) for input and (n_samples) for output
X_numpy, y_numpy = datasets.make_regression(n_samples=100, n_features=1, noise=20, random_state=4) # pyright: ignore
X_numpy: np.ndarray = X_numpy # shut pyright up
y_numpy: np.ndarray = y_numpy

# cast the data to pytorch tensors
# expected shapes are (n_samples, n_features) for input and output
X = torch.from_numpy(X_numpy.astype(np.float32))
y = torch.from_numpy(y_numpy.astype(np.float32))
y = y.view(y.shape[0], 1)

n_samples, n_features = X.shape

##
## model, loss. optimizer

# simple linear model of form f = wx + b
model = nn.Linear(n_features, 1)

learning_rate = 0.01

criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), learning_rate)

##
## training loop

epochs = 100
for epoch in range(epochs):
    # forward and loss
    y_pred = model(X)
    loss = criterion(y_pred, y)

    #backward and update params
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    if epoch % 10 == 0:
        print(f"epoch: {epoch}, loss = {loss.item():.5f}")

##
## plot results

predicted = model(X).detach().numpy() # detach makes a copy with disabled grad computation
plt.plot(X_numpy, y_numpy, 'bo') # regression problem, bo means blue points
plt.plot(X_numpy, predicted, 'r') # regression line, r means red line
plt.show()

##
