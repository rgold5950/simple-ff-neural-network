import numpy as np

from network import Network
from activationlayer import FCLayer
from activationlayer import ActivationLayer
from activationFunctions import tanh, tanh_prime
from loss_function import mse, mse_prime


#training data
x_train = np.array([[[0,0]], [[0,1]], [[1,0]], [[1,1]]])
y_train = np.array([[[0]], [[1]], [[1]], [[0]]])

#network
net = Network()
net.add(FCLayer(2,2))
net.add(ActivationLayer(tanh, tanh_prime))
net.add(FCLayer(2,1))
net.add(ActivationLayer(tanh, tanh_prime))


#train
net.use(mse, mse_prime)
net.fit(x_train, y_train, epochs=1000, learning_rate=0.1)

#test
out = net.predict(x_train)
print(out)