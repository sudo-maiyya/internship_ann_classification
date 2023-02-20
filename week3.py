#calling required libraries
import numpy as np
import pandas as pd


class ANN_Classification:
    def __init__ (self, hidden_Layer_Size = [100,], learning_Rate = 0.001, epochs = 10, X_val = None, Y_val = None ):
        self.hidden_Layer_Size = hidden_Layer_Size
        self.learning_Rate = learning_Rate
        self.epochs = epochs
       # self.activation_function = activation_function
        self.X_val = X_val
        self.Y_val = Y_val
        self.weights = None
        
    def Sigmoid(self, x):
      return 1 / (1 + np.exp(-x))

    def SigmoidDerivative(self, x):
      return np.multiply(x, 1-x)
            
    #function for forward propogation
    def forward_Prop(self, x, layers):
        activations, layer_input = [x], x
        for j in range(layers):
         # i = 
          #print("i = "+str(i))
          activation = self.Sigmoid(np.dot(layer_input, self.weights[j].T))
          activations.append(activation)
          layer_input = np.append(1, activation)
        
        return activations

    def back_prop(self, y, activations, layers):
      outputFinal = activations[-1]
      error = np.matrix(y - outputFinal) 
      
      # Error after 1 cycle
      for j in range(layers, 0, -1):
        currActivation = activations[j]
       
        if(j > 1):
          # Append previous
          prevActivation = np.append(1, activations[j-1])
        else:
          # First hidden layer
          prevActivation = activations[0]
       
        delta = np.multiply(error, self.SigmoidDerivative(currActivation))
        self.weights[j-1] += self.learning_Rate * np.multiply(delta.T, prevActivation)
         
        wc = np.delete(self.weights[j-1], [0], axis=1)
        error = np.dot(delta, wc) #current layer error
       
      return self.weights
    
    def initialize_Weight(self, layers):
      layer, self.weights = len(layers), []
      #for loop to intialize the weight randomly
      for i in range(1, layer):
        #assigning random weights
        w = [[np.random.uniform(-1, 1) for j in range(layers[i-1] + 1)]for k in range(layers[i])]
        self.weights.append(np.matrix(w))
    
      return self.weights
    
    #train function
    def train(self, X, y):
        layers_weights = len(self.weights)
        
        for i in range(len(self.X)):
          x, y = self.X[i], self.y[i]
          x = np.matrix(np.append(1, x))
          
          activations = self.forward_Prop(x, layers_weights)
          self.weights = self.back_prop(y, activations, layers_weights)
          
        return self.weights

    def fit(self, X, y):
        self.X = X
        self.y = y
        hidden_Layers = len(self.hidden_Layer_Size) - 1
        self.weights = self.initialize_Weight(self.hidden_Layer_Size)

        for epoch in range(1, self.epochs+1):
          weights = self.train(self.X, self.y)
         
          if(epoch % 1 == 0):
            print("Epoch {}".format(epoch))
            print("Training Accuracy:{}".format(self.Accuracy(self.X, self.y)))
            
            if self.X_val.any():
              print("Validation Accuracy:{}".format(self.Accuracy(self.X_val, self.Y_val)))
          
        return self.weights

        
    def Predict(self, x, acc = False):
     
      if acc == False:
        for i in range(len(X)):
          x, y = X[i], list(Y[i])

      layers = len(self.weights)
      item = np.append(1, x)

      # Forward prop.
      activations = self.forward_Prop(item, layers)
      
      Final_output = activations[-1].A1
      index = self.FindMaxActivation(Final_output)
      
      predicted = [0 for j in range(len(Final_output))]
      predicted[index] = 1 
      
      return predicted
      
    def FindMaxActivation(self, output):
      m, index = output[0], 0
      for i in range(1, len(output)):
        if(output[i] > m):
          m, index = output[i], i
      
      return index

    
    def Predicts(self, item):
        layers = len(self.weights)
        item = np.append(1, item)

        # Forward prop.
        activations = self.forward_Prop(item, layers)

        Final_output = activations[-1].A1
        index = self.FindMaxActivation(Final_output)

        y = [0 for j in range(len(Final_output))]
        y[index] = 1

        return y
    
    def Accuracy(self, X, Y):
        correct = 0

        for i in range(len(X)):
            x, y = X[i], list(Y[i])
            guess = self.Predicts(x)

            if (y == guess):
                # Right prediction
                correct += 1

        return correct / len(X)

    
        
