# Implementation Notes: Manual Backpropagation

## 1. Neural Network Implementation

The ECG heartbeat classification model was implemented using PyTorch tensor operations. Instead of relying on predefined neural network layers, the weights and biases were created and managed manually.

## 2. Forward Propagation

During forward propagation, the input ECG signal passes through the network's hidden layers. ReLU activation is applied to introduce non-linearity, while the output layer generates scores for the five heartbeat categories.

## 3. Manual Backpropagation

Backpropagation calculates the gradients of the loss function with respect to the model parameters. These gradients indicate how the weights and biases should be adjusted to reduce classification error.

The chain rule is used to calculate gradients layer by layer. This provides a practical understanding of how neural networks learn from training data.

## 4. Gradient Verification

A gradient-checking procedure was used to compare analytical gradients with numerical approximations. The small differences observed during verification support the correctness of the manually implemented gradient calculations.

## 5. Training and Optimization

The model was trained using manual momentum-based gradient descent with weight decay. Three learning rates (0.01, 0.1, and 0.5) were evaluated over 30 epochs to compare training performance.

## 6. Conclusion

Implementing these operations manually provided a deeper understanding of neural network training, gradient calculations, parameter updates, and the optimization process.
