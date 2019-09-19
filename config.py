"""
config.py

configurations used throughout the project. Manipulating these parameters can have drastic changes on the outcome of
the models.
"""

ADAM_LR = 0.1  # (Float) Drastic increase the learning rate since optima are situated at the edges [def=0.2]
BATCH_SIZE = 128  # (Integer) Training batch size [def=128]
CURATE_BATCH = 4  # (Integer) Number of samples curated each iteration of network_2's training [def=4]
THRESHOLD = 0.5  # (Float) Threshold (0..1) that determines the test between the two categories [def=0.5]
UNCERTAINTY_STEP = 0.1  # (Float) Narrow each epoch the uncertainty interval with this step-size [def=0.1]
