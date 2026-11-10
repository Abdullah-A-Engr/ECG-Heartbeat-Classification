
# ECG Heartbeat Classification — Final Analysis

## 1. Test Performance

Test accuracy: 97.73%

Balanced accuracy: 81.77%

Always-N baseline: 82.76%

The model improves over the always-N baseline by
14.97 percentage points.

## 2. Per-Class Accuracy

| Class | Heartbeat | Accuracy |
|---|---|---:|
| 0 | N | 99.41% |
| 1 | S | 64.03% |
| 2 | V | 94.89% |
| 3 | F | 53.09% |
| 4 | Q | 97.45% |

## 3. Confusion Matrix

Rows represent actual classes and columns represent predicted classes.

| Actual / Predicted | N | S | V | F | Q |
|---|---:|---:|---:|---:|---:|
| N | 18012 | 19 | 60 | 2 | 25 |
| S | 181 | 356 | 16 | 0 | 3 |
| V | 62 | 2 | 1374 | 7 | 3 |
| F | 47 | 0 | 29 | 86 | 0 |
| Q | 30 | 0 | 11 | 0 | 1567 |

## 4. Interpretation

The weakest class is F,
with 53.09% accuracy.
This indicates difficulty distinguishing some minority-class heartbeat patterns.

The model achieves high overall accuracy, but balanced accuracy is lower
because the dataset is imbalanced. The always-N baseline is already
82.76%, so balanced accuracy and
per-class performance are important additional measures.

## 5. Improvement

A suitable future improvement is to use class-weighted loss or
oversampling for the rare fusion (F) class. This may improve minority-class
recognition, although it would need a separate validation experiment.
