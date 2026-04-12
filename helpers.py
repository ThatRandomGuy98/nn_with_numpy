import numpy as np
import nnfs
from nnfs.datasets import spiral_data


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