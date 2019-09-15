"""
config.py

configurations used throughout the project. Manipulating these parameters can have drastic changes on the outcome of
the models.

Config overview:
 * MODEL - Model (Sentiment-Learner) specific parameters
"""

# ------------------------------------------------------> MODEL <----------------------------------------------------- #

BATCH_SIZE = 128  # (Integer) Training batch size
THRESHOLD = 0.5  # (Float) Threshold (0..1) that determines the test between the two categories
VERSION = 0  # (Integer) 0: No versioning | >0: Model version number (including tokenizer, meta-data, ...) [def=0]
