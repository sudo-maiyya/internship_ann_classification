import numpy as np
import pandas as pd

class ANN_Classification:
    def __init__(self, hidden_Layer_Size=[100,], learning_Rate=0.001, epoch=10, X_val=None, Y_val=None):
        self.hidden_Layer_Size = hidden_Layer_Size
        self.learning_Rate = learning_Rate
        self.epoch = epoch
       # self.activation_function = activation_function
        self.X_val = X_val
        self.Y_val = Y_val

    def Sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def SigmoidDerivative(self, x):
        return np.multiply(x, 1-x)

    # function for forward propogation
    def forward_Prop(self, x, weights, layers):
        activations, layer_input = [x], x
        for j in range(layers):
         # i =
            # print("i = "+str(i))
            activation = self.Sigmoid(np.dot(layer_input, weights[j].T))
            activations.append(activation)
            layer_input = np.append(1, activation)

        return activations

    def back_prop(self, y, activations, weights, layers):
        outputFinal = activations[-1]
        error = np.matrix(y - outputFinal)

        # Error after 1 cycle
        for j in range(layers, 0, -1):
            currActivation = activations[j]

            if (j > 1):
                # Append previous
                prevActivation = np.append(1, activations[j-1])
            else:
                # First hidden layer
                prevActivation = activations[0]

            delta = np.multiply(error, self.SigmoidDerivative(currActivation))
            weights[j-1] += self.learning_Rate * \
                np.multiply(delta.T, prevActivation)

            wc = np.delete(weights[j-1], [0], axis=1)
            error = np.dot(delta, wc)  # current layer error

        return weights

    def initialize_Weight(self, layers):
        layer, weights = len(layers), []
        # for loop to intialize the weight randomly
        for i in range(1, layer):
            # assigning random weights
            w = [[np.random.uniform(-1, 1) for j in range(layers[i-1] + 1)]
                 for k in range(layers[i])]
            weights.append(np.matrix(w))

        return weights

    # train function
    def train(self, X, y, weights):
        layers_weights = len(weights)

        for i in range(len(self.X)):
            x, y = self.X[i], self.y[i]
            x = np.matrix(np.append(1, x))

            activations = self.forward_Prop(x, weights, layers_weights)
            weights = self.back_prop(y, activations, weights, layers_weights)

        return weights

    def fit(self, X, y):
        self.X = X
        self.y = y
        hidden_Layers = len(self.hidden_Layer_Size) - 1
        weights = self.initialize_Weight(self.hidden_Layer_Size)

        for epoch in range(1, self.epoch+1):
            weights = self.train(self.X, self.y, weights)

            if (epoch % 10 == 0):
                print("Epoch {}".format(epoch))
                print("Training Accuracy:{}".format(
                    self.Accuracy(self.X, self.y, weights)))

                if self.X_val.any():
                    print("Validation Accuracy:{}".format(
                        self.Accuracy(self.X_val, self.Y_val, weights)))

        return weights

    def Predict(self, item, weights):
        layers = len(weights)
        item = np.append(1, item)

        # Forward prop.
        activations = self.forward_Prop(item, weights, layers)

        Final_output = activations[-1].A1
        index = self.FindMaxActivation(Final_output)

        y = [0 for j in range(len(Final_output))]
        y[index] = 1

        return y

    def FindMaxActivation(self, output):
        m, index = output[0], 0
        for i in range(1, len(output)):
            if (output[i] > m):
                m, index = output[i], i

        return index

    def Accuracy(self, X, Y, weights):
        correct = 0

        for i in range(len(X)):
            x, y = X[i], list(Y[i])
            guess = self.Predict(x, weights)

            if (y == guess):
                # Right prediction
                correct += 1

        return correct / len(X)
