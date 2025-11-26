"""
13/november/2025
Little pet project that serves no purpose other than understanding neural networks bit better
"""

import matplotlib.pyplot as plt
import numpy as np
import nnfs
from nnfs.datasets import spiral_data

np.random.seed(22)
nnfs.init()


class DenseLayer:
    def __init__(self, n_inputs, n_neurons) -> None:
        self.weights = 0.1 * np.random.randn(n_inputs, n_neurons)   # small random weights (can be positive or negative)
        self.biases = np.zeros((1, n_neurons))

    def forward(self, inputs) -> None:
        self.inputs = inputs
        self.output = np.dot(inputs, self.weights) + self.biases

    def backward(self, dvalues) -> None:
        """
        dvalues: gradient of the loss w.r.t. this layer's output
        Computes gradients for weights, biases, and inputs.
        """
        self.dweights = np.dot(self.inputs.T, dvalues)  # gradient w.r.t. weights: X^T * dvalues
        self.dbiases = np.sum(dvalues, axis=0, keepdims=True)   # gradient w.r.t. biases: sum over samples
        self.dinputs = np.dot(dvalues, self.weights.T)  # gradient w.r.t. inputs: dvalues * W^T


class ReLUActivation:
    def forward(self, inputs) -> None:
        self.inputs = inputs
        self.output = np.maximum(0, inputs) # store inputs for backprop (to know where we were <= 0)

    def backward(self, dvalues) -> None:
        """
        For ReLU: gradient passes through where input > 0,
        and is zero where input <= 0.
        """
        self.dinputs = dvalues.copy()
        self.dinputs[self.inputs <= 0] = 0


class SoftmaxActivation:
    def forward(self, inputs) -> None:
        # numeric stability trick: subtract max per sample
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        self.output = probabilities
        


class Loss:
    def calculate_loss(self, y_pred, y_true) -> float:
        sample_losses = self.forward(y_pred, y_true)
        data_loss = np.mean(sample_losses)
        return data_loss


class CategoricalCrossentropy(Loss):
    def forward(self, y_pred, y_true) -> float:
        """
        y_pred: probabilities from softmax, shape (n_samples, n_classes)
        y_true: class indices (shape (n_samples,)) OR one-hot (n_samples, n_classes)
        """
        n_samples = len(y_pred)
        y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)

        if len(y_true.shape) == 1:  # sparse labels: pick the prob for the correct class
            correct_confidence = y_pred_clipped[range(n_samples), y_true]
            
        elif len(y_true.shape) == 2:    # one-hot: multiply and sum
            correct_confidence = np.sum(y_pred_clipped * y_true, axis=1)
            
        else:
            raise ValueError("y_true must be 1D (sparse) or 2D (one-hot)")

        negative_log_likelihood = -np.log(correct_confidence)
        return negative_log_likelihood

    def backward(self, dvalues, y_true):
        """
        Combined Softmax + Cross-Entropy derivative.
        dvalues: output of softmax (probabilities), shape (n_samples, n_classes)
        y_true: class indices or one-hot.
        Result: gradient of loss w.r.t. pre-softmax layer output.
        """
        n_samples = len(dvalues)

        # If labels are one-hot, turn them into class indices
        if len(y_true.shape) == 1:
            y_true_indices = y_true
        elif len(y_true.shape) == 2:
            y_true_indices = np.argmax(y_true, axis=1)
        else:
            raise ValueError("y_true must be 1D (sparse) or 2D (one-hot)")

        
        self.dinputs = dvalues.copy()   # Copy so we don't modify the softmax output in-place
        self.dinputs[range(n_samples), y_true_indices] -= 1     # Subtract 1 from the probabilities of the correct class
        self.dinputs = self.dinputs / n_samples     # Average across samples


X, y = spiral_data(samples=100, classes=3)
dense_1 = DenseLayer(n_inputs=2, n_neurons=3)
activation_1 = ReLUActivation()
dense_2 = DenseLayer(n_inputs=3, n_neurons=3)
activation_2 = SoftmaxActivation()

loss_fn = CategoricalCrossentropy()
learning_rate = 1.0
EPOCHS = 1000


for epoch in range(EPOCHS):

    dense_1.forward(X)
    activation_1.forward(dense_1.output)
    dense_2.forward(activation_1.output)
    activation_2.forward(dense_2.output)

    loss = loss_fn.calculate_loss(activation_2.output, y)
    predictions = np.argmax(activation_2.output, axis=1)
    accuracy = np.mean(predictions == y)

    if epoch % 100 == 0 or epoch == EPOCHS - 1:
        print(f"Epoch {epoch:4d} | Loss: {loss:.4f} | Acc: {accuracy:.4f}")

    loss_fn.backward(activation_2.output, y)
    dense_2.backward(loss_fn.dinputs)
    activation_1.backward(dense_2.dinputs)
    dense_1.backward(activation_1.dinputs)

    dense_1.weights -= learning_rate * dense_1.dweights
    dense_1.biases  -= learning_rate * dense_1.dbiases
    dense_2.weights -= learning_rate * dense_2.dweights
    dense_2.biases  -= learning_rate * dense_2.dbiases

print(f"--- Final predictions (first 10): {predictions[:10]} ---")