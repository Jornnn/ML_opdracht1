import random
import math

class nodeModel:
    def __init__(self, input_dim, output_dim):
        self.W = [[random.uniform(-1, 1) for _ in range(output_dim)] for _ in range(input_dim)]

    def softmax(self, x):
        max_val = max(x)
        exps = [math.exp(i - max_val) for i in x]
        sum_exps = sum(exps)
        return [j / sum_exps for j in exps]

    def dot(self, a, b):
        result = []
        for i in range(len(a)):
            row = []
            for j in range(len(b[0])):
                s = 0
                for k in range(len(b)):
                    s += a[i][k] * b[k][j]
                row.append(s)
            result.append(row)
        return result

    def transpose(self, matrix):
        return list(map(list, zip(*matrix)))


    def forward(self, X):
        z = self.dot(X, self.W)

        return [self.softmax(row) for row in z]

    def train(self, X, Y, epochs=100, lr=0.2):
        for epoch in range(epochs):
            output = self.forward(X)
            MSEloss = sum(
                sum((y[j] - o[j]) ** 2 for j in range(len(y)))
                for y, o in zip(Y, output)
            ) / len(Y)

            dz = [[2 * (o[j] - y[j]) for j in range(len(o))] for o, y in zip(output, Y)]
            dW = self.dot(self.transpose(X), dz)

            for i in range(len(self.W)):
                for j in range(len(self.W[0])):
                    self.W[i][j] -= lr * dW[i][j]
            

            if epoch % 1 == 0:
                print(f"Epoch {epoch} - MSELoss: {MSEloss:.4f}")

    # def predict(self, X):
    #     output = self.forward(X)
    #     return [row.index(max(row)) for row in output]
    
    def predict(self, X):
        output = self.forward(X)
        predictions = []
        for row in output:
            predicted_class = row.index(max(row))
            confidence = max(row)  # Softmax value for predicted class
            predictions.append((predicted_class, confidence))
        return predictions

