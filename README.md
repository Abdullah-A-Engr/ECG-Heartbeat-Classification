# ECG Heartbeat Classification Using PyTorch

## Project Overview

This project implements a neural network from scratch using PyTorch to classify ECG heartbeat signals into five categories. The model is trained and evaluated using the MIT-BIH Arrhythmia dataset.

The project focuses on understanding neural network fundamentals, manual backpropagation, weight initialization, optimization, and model evaluation.

## Heartbeat Classes

| Label | Class |
|---|---|
| N | Normal |
| S | Supraventricular |
| V | Ventricular |
| F | Fusion |
| Q | Unknown/Paced |

## Implementation

The neural network was implemented manually using PyTorch tensor operations.

Key components include:
- Manual weight and bias initialization using He initialization.
- Manual forward propagation.
- Manual backpropagation.
- Cross-entropy loss using `torch.logsumexp`.
- Momentum-based gradient descent with weight decay.
- Learning-rate comparison using 0.01, 0.1, and 0.5.
- Training for 30 epochs per learning rate.
- Manual evaluation metrics and confusion matrix.

## Experimental Results

The best-performing configuration used a learning rate of **0.5**, with the best validation checkpoint at epoch **22**.

| Metric | Result |
|---|---:|
| Test Accuracy | 97.73% |
| Balanced Accuracy | 81.77% |
| Best Learning Rate | 0.5 |
| Best Epoch | 22 |
| Always-N Baseline | 82.76% |

### Per-Class Accuracy

| Class | Accuracy |
|---|---:|
| N - Normal | 99.41% |
| S - Supraventricular | 64.03% |
| V - Ventricular | 94.89% |
| F - Fusion | 53.09% |
| Q - Unknown/Paced | 97.45% |

The results show strong overall classification performance. However, the lower accuracy for the Fusion class highlights the impact of class imbalance.

## Visual Results

The project includes:
- Training and validation accuracy curves.
- Training and validation loss curves.
- Learning-rate comparison.
- Test confusion matrix.
- Example ECG heartbeat signals.
- Streamlit frontend screenshot.

## Frontend

A Streamlit application was developed to demonstrate the trained model. It allows users to select a test sample or upload an ECG signal and view the predicted heartbeat category and model scores.

**Note:** This project is educational and is not intended for clinical diagnosis.

## Dataset

The project uses the MIT-BIH Arrhythmia dataset.

The dataset files are not included in this GitHub repository because of their large size. They should be placed in the appropriate `data/` directory before running the project.

## Technologies Used

- Python
- PyTorch
- NumPy
- Pandas
- Matplotlib
- Streamlit

## Author

ECG Heartbeat Classification Project — PyTorch from Scratch.
int, test metrics, or confusion matrix is claimed here. Run `python src/train.py` to completion, followed by `python src/test_model.py`, to generate these items.
