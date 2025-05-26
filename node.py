import random
import math

class nodeModel:
    def __init__(self, input_dim, output_dim):
        assert output_dim == 2, "This version only supports two output classes."
        self.input_dim = input_dim
        # Separate weights and bias for each class
        self.W_0 = [random.uniform(-1, 1) for _ in range(input_dim)]
        self.b_0 = random.uniform(-1, 1)

        self.W_1 = [random.uniform(-1, 1) for _ in range(input_dim)]
        self.b_1 = random.uniform(-1, 1)

    def softmax(self, scores):
        max_score = max(scores)
        exps = [math.exp(s - max_score) for s in scores]
        sum_exps = sum(exps)
        return [e / sum_exps for e in exps]

    def forward(self, x):
        # Manually compute the scores
        score_0 = sum(x[i] * self.W_0[i] for i in range(self.input_dim)) + self.b_0
        score_1 = sum(x[i] * self.W_1[i] for i in range(self.input_dim)) + self.b_1
        return self.softmax([score_0, score_1])

    def train(self, X, Y, epochs=12, lr=0.25):
        for epoch in range(epochs):
            total_loss = 0
            for x, y in zip(X, Y):  # y is expected as [0, 1] or [1, 0]
                output = self.forward(x)
                loss = sum((y[i] - output[i]) ** 2 for i in range(2))
                total_loss += loss

                # Compute gradients and update weights for both classes
                for i in range(self.input_dim):
                    grad_0 = 2 * (output[0] - y[0]) * output[0] * (1 - output[0]) * x[i]
                    grad_1 = 2 * (output[1] - y[1]) * output[1] * (1 - output[1]) * x[i]

                    self.W_0[i] -= lr * grad_0
                    self.W_1[i] -= lr * grad_1

                self.b_0 -= lr * 2 * (output[0] - y[0]) * output[0] * (1 - output[0])
                self.b_1 -= lr * 2 * (output[1] - y[1]) * output[1] * (1 - output[1])

            print(f"Epoch {epoch} - MSELoss: {total_loss / len(X):.4f}")

    def predict(self, X):
        predictions = []
        for x in X:
            output = self.forward(x)
            label = output.index(max(output))
            confidence = max(output)
            predictions.append((label, confidence))
        return predictions
