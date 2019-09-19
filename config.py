"""
config.py

configurations used throughout the project. Manipulating these parameters can have drastic changes on the outcome of
the models.
"""

BATCH_SIZE = 128  # (Integer) Training batch size [def=128]
CURATE_BATCH = 32  # (Integer) Number of samples curated each iteration of network_2's training [def=32]
THRESHOLD = 0.5  # (Float) Threshold (0..1) that determines the test between the two categories [def=0.5]
UNCERTAIN_MAX = 0.8  # (Float) Maximum threshold to indicate 'uncertain' samples [def=0.8]
UNCERTAIN_MIN = 0.2  # (Float) Minimum threshold to indicate 'uncertain' samples [def=0.2]
VERSION = 0  # (Integer) 0: No versioning | >0: Model version number (including tokenizer, meta-data, ...) [def=0]
