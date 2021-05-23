import numpy as np
from network_wMBGD import NetworkMBGD
from network import Network
from activationlayer import FCLayer
from activationlayer import ActivationLayer
from activationFunctions import tanh, tanh_prime
from loss_function import mse, mse_prime

from keras.datasets import mnist
from keras.utils import np_utils
import time

start_time = time.time()
"-----------------here we are testing mini-batch gradient descent-----------------------------------------"
#load MNIST from server
(x_train, y_train), (x_test, y_test) = mnist.load_data()

#training data : 6000 samples
#reshape and normalize input data
x_train = x_train.reshape(x_train.shape[0], 1, 28*28)
x_train = x_train.astype('float32')
x_train /= 255
#encode output which is a number in range [0,9] into a vector of size 10
#e.g. number 3 will become [0,0,0,1,0,0,0,0,0,0]
y_train = np_utils.to_categorical(y_train)

#same for test data : 1000 samples
x_test = x_test.reshape(x_test.shape[0], 1, 28*28)
x_test = x_test.astype('float32')
x_test /= 255
y_test = np_utils.to_categorical(y_test)

#Network
net = NetworkMBGD()
net.add(FCLayer(28*28,100)) #input_shape=(1,28*28) output_shape = (1,100)
net.add(ActivationLayer(tanh, tanh_prime))
net.add(FCLayer(100,50))  # input_shape=(1,100) output_shap=(1,50)
net.add(ActivationLayer(tanh, tanh_prime))
net.add(FCLayer(50,10))
net.add(ActivationLayer(tanh, tanh_prime))

#since we are using mini-batch, we can train on all the samples
net.use(mse,mse_prime)
net.fit(x_train, y_train, epochs=35, learning_rate=0.1)

# test on 3 samples
out = net.predict(x_test[0:3])
print("\n")
print("predicted values : ")
print(out, end="\n")
print("true values : ")
print(y_test[0:3])
print("----%s seconds----" % (time.time() - start_time))



"--------------------here we test standard gradient descent----------------------"
start_time = time.time()
#Network
net2 = Network()
net2.add(FCLayer(28*28,100)) #input_shape=(1,28*28) output_shape = (1,100)
net2.add(ActivationLayer(tanh, tanh_prime))
net2.add(FCLayer(100,50))  # input_shape=(1,100) output_shap=(1,50)
net2.add(ActivationLayer(tanh, tanh_prime))
net2.add(FCLayer(50,10))
net2.add(ActivationLayer(tanh, tanh_prime))

#train on 1000 samples
#as we didn't implement mini-batch GD, training will be slow if we update at each iteration on 60000 samples
net2.use(mse,mse_prime)
net2.fit(x_train[0:1000], y_train[0:1000], epochs=35, learning_rate=0.1)

# test on 3 samples
out2 = net2.predict(x_test[0:3])
print("\n")
print("predicted values with standard GD: ")
print(out2, end="\n")
print("true values : ")
print(y_test[0:3])

print('Difference between standard - minibatchgd: ')
print(out-out2)
print("----%s seconds----" % (time.time() - start_time))

