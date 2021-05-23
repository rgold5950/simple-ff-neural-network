#based on code from Omar Aflak: https://towardsdatascience.com/math-neural-network-from-scratch-in-python-d6da9f29ce65
import numpy as np
class NetworkMBGD:
    def __init__(self):
        self.layers = []
        self.loss = None
        self.loss_prime = None

    #add layer to network
    def add(self, layer):
        self.layers.append(layer)

    #set loss to use
    def use(self, loss, loss_prime):
        self.loss = loss
        self.loss_prime = loss_prime

    #predict output for given input
    def predict(self, input_data):
        #sample dimensions first
        samples = len(input_data)
        result = []

        #run network over all samples
        for i in range(samples):
            # forward propagation
            output = input_data[i]
            for layer in self.layers:
                output = layer.forward_propagation(output)
            result.append(output)

        return np.array(result)

    def generate_next_minibatch(self, inputs, start_idx, batch_size):
        if start_idx >= inputs.shape[0]:
            return "index has exceeded sample size"
        end_idx = min(start_idx+batch_size, inputs.shape[0])
        return start_idx, end_idx


    #train the network
    def fit(self, x_train, y_train, epochs, learning_rate):
        #sample dimension first
        samples = len(x_train)

        batch_size = 50
        curr_idx = 0

        #training loop
        for i in range(epochs):
            err = 0
            #iterating through samples; this time we just iterate through batches of mini-batches of samples instead of going through all of them each epoch
            for j in range(curr_idx,min(curr_idx+50, len(x_train))):
                # forward propagation
                output = x_train[j]
                for layer in self.layers:
                    output = layer.forward_propagation(output)

                #compute loss (for display purpose only)
                err += self.loss(y_train[j], output)

                #backward propagation

                #dE/dY term here
                error = self.loss_prime(y_train[j], output)
                for layer in reversed(self.layers):
                    error = layer.backward_propagation(error, learning_rate)

            #calculate average error on all samples
            err /= samples
            print('epoch %d/%d  error=%f' % (i+1, epochs, err))

            if curr_idx <= len(x_train):
                curr_idx += batch_size
            else:
                curr_idx = 0
        return print("final error is: " + str(err))
