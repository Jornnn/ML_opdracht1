import random
import math
import numpy as np
import matplotlib.pyplot as plt

class nodeModel:
    # Create weight per layer
    def __init__(self, input_dim, output_dim, hidden_dim):
        self.W1 = [[random.uniform(-1, 1) for _ in range(hidden_dim)] for _ in range(input_dim)]
        self.W2 = [[random.uniform(-1, 1) for _ in range(output_dim)] for _ in range(hidden_dim)]

    # Sigmoid made using this source: https://www.digitalocean.com/community/tutorials/sigmoid-activation-function-python
    def sigmoid(self, x):
        return 1.0/(1 + math.exp(-x))

    # Softmax made using this source: https://www.loekvandenouweland.com/content/softmax-with-plain-python.html
    def softmax(self, x):
        max_val = max(x)
        exps = [math.exp(i - max_val) for i in x]
        sum_exps = sum(exps)
        return [j / sum_exps for j in exps]
    
    # Dot product matrix calc
    def dot(self, a, b):
        # VERSION 2.1
        # result = []
        # for i in range(len(a)):
        #     row = []
        #     for j in range(len(b[0])):
        #         s = 0
        #         for k in range(len(b)):
        #             s += a[i][k] * b[k][j]
        #         row.append(s)
        #     result.append(row)

        # VERSION 2.2
        result = np.dot(a, b)
        return result

    #Transpose matrix calc
    def transpose(self, matrix):
        # VERSION 2.1
        # return list(map(list, zip(*matrix)))
        
        # VERSION 2.2
        return np.transpose(matrix)


    def forward(self, X):
        self.Z1 = self.dot(X, self.W1)  # Pre-activation for hidden layer
        self.A1 = [[self.sigmoid(z) for z in row] for row in self.Z1]  # Sigmoid activation

        self.Z2 = self.dot(self.A1, self.W2)  # Pre-activation for output layer
        output = [self.softmax(row) for row in self.Z2]  # Softmax activation
        return output
    
    # Draw graph of the loss
    def draw_result(self, MSEloss, title):
        epochs = list(range(len(MSEloss)))  # x-as: 0, 1, ..., n
        plt.plot(epochs, MSEloss, '-b', label='loss', color= "red")

        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend(loc='upper left')
        plt.title(title)

        plt.savefig(title + ".png")  # Save before show
        plt.show()


    # Trains the model, source: https://www.geeksforgeeks.org/backpropagation-in-neural-network/
    def train(self, X, Y, epochs=1000, lr=0.01):
        lst_loss = []
        for epoch in range(epochs):
            output = self.forward(X)

            # Loss calculation (MSE)
            MSEloss = sum(
                sum((y[j] - o[j]) ** 2 for j in range(len(y)))
                for y, o in zip(Y, output)
            ) / len(Y)

            # Output layer
            dZ2 = [[2 * (o[j] - y[j]) for j in range(len(o))] for o, y in zip(output, Y)]
            dW2 = self.dot(self.transpose(self.A1), dZ2)

            # Hidden layer (backprop)
            dA1 = self.dot(dZ2, self.transpose(self.W2))
            dZ1 = [
                [dA1[i][j] * self.A1[i][j] * (1 - self.A1[i][j]) for j in range(len(dA1[0]))]
                for i in range(len(dA1))
            ]
            dW1 = self.dot(self.transpose(X), dZ1)

            # Update weights
            for i in range(len(self.W2)):
                for j in range(len(self.W2[0])):
                    self.W2[i][j] -= lr * dW2[i][j]
            for i in range(len(self.W1)):
                for j in range(len(self.W1[0])):
                    self.W1[i][j] -= lr * dW1[i][j]

            if epoch % 1 == 0:
                lst_loss.append(MSEloss)
                print(f"Epoch {epoch} - MSELoss: {MSEloss:.4f}")
                        # Stop early if loss is low enough

            if MSEloss < 0.01:
                break
        return lst_loss


    # Returns the prediction and confidence level of the model
    def predict(self, X):
        output = self.forward(X)
        predictions = []
        for row in output:
            predicted_class = row.index(max(row))
            confidence = max(row)  # Softmax value for predicted class
            predictions.append((predicted_class, confidence))
        return predictions
    


