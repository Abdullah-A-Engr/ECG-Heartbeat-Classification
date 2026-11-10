# Experiment status

- Dataset validation and exact seeded split: completed.
- Example waveform figure: generated.
- Architecture parameter count: 32,645 (verified).
- Output-layer gradient check: completed; max absolute error W3 = 2.98e-08, b3 = 2.98e-08.
- Full training: incomplete in the current execution environment. The CPU run reached learning rate 0.1, epoch 20 before the execution time limit. Do not treat partial checkpoints as the selected final model.
- Final test evaluation and frontend screenshot: pending completion of full training and Streamlit installation.
