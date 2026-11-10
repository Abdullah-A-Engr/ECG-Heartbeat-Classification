# Can a Neural Network Learn to Read Heartbeats? Building an ECG Classifier from Scratch in PyTorch

## Introduction
An electrocardiogram (ECG) records the electrical activity of the heart over time. Individual beats have characteristic waveform patterns, and changes in those patterns can be associated with different heartbeat categories. In this project, we built a five-class classifier for short ECG beat segments using a fully connected neural network implemented in PyTorch. The emphasis was not only on obtaining predictions, but on understanding the mechanics of a neural network: parameter initialization, forward propagation, cross-entropy loss, backpropagation, gradient verification, and a manually implemented optimization loop.

This is an educational machine-learning project. Its predictions are not intended for clinical use and cannot replace assessment by qualified healthcare professionals.

## Dataset and preparation
The experiment uses the MIT-BIH heartbeat categorization CSV files. Each row contains 187 signal samples followed by one integer label. The five categories are N (normal), S (supraventricular ectopic), V (ventricular ectopic), F (fusion), and Q (unknown or paced). The supplied training file contains 87,554 rows and the separate test file contains 21,892 rows. The signal samples are normalized, and the provided sampling rate is 125 Hz.

We kept the test file separate until final evaluation. A seeded NumPy permutation (seed 42) divided the training file into 90% training and 10% validation data. Signal arrays were converted to `torch.float32` and labels to `torch.long`. We checked dimensions, signal range, label values, and class frequencies before training. [Insert the generated data summary table here.]

**Figure 1.** Add `results/figures/example_beats.png`, showing one example waveform from each class. The examples demonstrate that each category has its own distribution of waveform shapes, while some categories can look similar. This visual overlap is one reason class-wise evaluation matters.

## Network architecture
The model is a fully connected network with 187 input features, two hidden layers containing 128 and 64 units, and five output logits. ReLU is applied to the hidden layers using `torch.clamp(z, min=0)`. The output layer is left linear because the loss function operates directly on logits.

All six weight and bias tensors are explicit `nn.Parameter` objects. The architecture contains 32,645 trainable parameters. We initialized weights with He scaling, using a standard deviation of the square root of two divided by the layer's fan-in, and initialized biases to zero. Initializing every weight to zero would preserve symmetry: neurons in the same layer would receive identical gradients and fail to learn distinct features. He initialization is suited to ReLU networks because it helps maintain activation and gradient scale through the layers.

## Loss and gradient verification
We implemented mean softmax cross-entropy directly from logits using `torch.logsumexp`. This avoids using a prebuilt cross-entropy loss. Before training, we computed gradients on the first 64 training examples and independently derived the output-layer weight and bias gradients. The maximum absolute differences between the manual gradients and autograd gradients are recorded in the generated gradient-check output. [Insert the measured differences from your run.]

## Training strategy
The training loop uses minibatches of 128 examples for 30 epochs. Each epoch shuffles the training indices. We compared learning rates 0.01, 0.1, and 0.5 using the same initialization seed and validation split. Momentum was set to 0.9 and weight decay to 0.0001. The update maintains a velocity for each parameter; momentum blends the current gradient with the previous velocity, making updates less sensitive to noisy minibatch gradients. Weight decay adds a small parameter-proportional term to weight gradients, discouraging excessively large weights. Biases are not decayed.

**Figure 2.** Insert `results/figures/learning_rate_comparison.png` and identify the learning rate with the lowest validation loss. **Figure 3.** Insert the chosen run's loss and accuracy curves. Report the best epoch and training/validation metrics at that point. A widening gap between training and validation curves may indicate overfitting; persistently high losses may indicate underfitting.

## Test results
After selecting the learning rate and saving the checkpoint with the lowest validation loss, we loaded the selected checkpoint and evaluated the test set once. The evaluation reports overall accuracy, balanced accuracy, per-class accuracy, and a 5-by-5 confusion matrix. The always-N baseline is calculated from the test class distribution; the supplied counts imply 18,118 normal examples among 21,892, or approximately 82.76%. [Insert the actual measured test metrics and confusion matrix here.]

The weakest class and most frequent off-diagonal confusion should be identified from the generated metrics. Confusions can arise because distinct heartbeat categories may share similar waveform morphology, while the imbalance in the data gives the network many more examples of common categories. Class F has only 641 examples in the training set. A reasonable next experiment would be class-weighted loss or carefully controlled oversampling for F, with model selection based only on validation data.

## Conclusion
This assignment demonstrates the core mechanics behind a neural network rather than relying on high-level training abstractions. The implementation explicitly defines parameters, computes logits, evaluates cross-entropy, checks gradients, and applies momentum-based updates. The final conclusion should be based on the generated validation curves and held-out test metrics, not on an assumed accuracy. Once the result fields and figures have been inserted, publish this article with the project repository link.

---
**Before publication:** replace all bracketed instructions with measured values, add the generated figures and tables, and verify the article falls within the required 800–1,500 words.

## Reproducibility and limitations

Reproducibility is important in a supervised-learning experiment because a model's performance can change when the split, initialization, or optimization settings change. For that reason, this implementation fixes the validation permutation seed and resets the model initialization seed for each learning-rate trial. The training histories are saved as CSV files so that the figures can be regenerated and the selected epoch can be checked independently. The final test file is not used to choose the learning rate or checkpoint; it is reserved for the final report.

There are also limitations to consider when interpreting the results. The model receives a fixed-length, one-dimensional beat segment and does not use patient-level metadata, temporal context from neighboring beats, or clinical information. The fully connected architecture treats the samples as a vector rather than explicitly modeling local temporal patterns. In addition, the class distribution is imbalanced, so a high overall accuracy can coexist with low recall for a rare category. Balanced accuracy and the confusion matrix are therefore important alongside the overall score.

The Streamlit interface makes the model easier to demonstrate: a user can select a row from the held-out test CSV or upload a row containing 187 signal values. The interface plots the waveform and displays the predicted category with the five model probability scores. These scores express the model's relative output for the supplied input; they are not calibrated medical probabilities and must not be interpreted as a diagnosis.

For a future version, we would compare the fully connected baseline with a one-dimensional convolutional model, investigate class-weighted training, and assess calibration on a separate validation set. Any such extension would need to preserve the held-out test set until the final comparison. The present assignment deliberately focuses on a transparent implementation of the basic neural-network training process, so these extensions are recommendations rather than part of the reported experiment.


## Actual Test Results


## Final Test Results

- Test accuracy: **97.73%**
- Balanced accuracy: **81.77%**
- Always-N baseline: **82.76%**

### Per-class accuracy

- N: 99.41%
- S: 64.03%
- V: 94.89%
- F: 53.09%
- Q: 97.45%

### Interpretation

The model achieves high overall test accuracy, but performance differs
across heartbeat classes. Fusion (F) is the weakest class, reflecting
the challenge of learning from underrepresented categories.

### Figures

- `results/figures/example_beats.png`
- `results/figures/learning_rate_comparison.png`
- `results/figures/loss_curves.png`
- `results/figures/accuracy_curves.png`
- `results/figures/confusion_matrix.png`
- `results/figures/frontend_screenshot.png`
