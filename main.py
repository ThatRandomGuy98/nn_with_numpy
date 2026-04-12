"""
13/november/2025
Little pet project that serves no purpose other than understanding neural networks bit better
"""

import numpy as np
import nnfs
from nnfs.datasets import spiral_data

from helpers import (
    DenseLayer, ReLUActivation, SoftmaxActivation,
    Loss, CategoricalCrossentropy
    )

np.random.seed(22)
nnfs.init()


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