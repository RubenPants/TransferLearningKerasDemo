"""
fetch_data.py

Fetch the MNIST dataset in its raw format.
"""
import mnist
import pickle

from myutils import *


def fetch_test():
    """
    Fetch from the test set both the images and the labels, and store in the ./raw/ directory.
    """
    # Images
    img = mnist.test_images()
    save_to_raw(img, "test_images")
    
    # Labels
    labels = mnist.test_labels()
    save_to_raw(labels, "test_labels")


def fetch_train():
    """
    Fetch from the training set both the images and the labels, and store in the ./raw/ directory.
    """
    # Images
    img = mnist.train_images()
    save_to_raw(img, "train_images")
    
    # Labels
    labels = mnist.train_labels()
    save_to_raw(labels, "train_labels")


def save_to_raw(file, name):
    """
    Save a file as a pickle file to the raw subdirectory under the given name.

    :param file: File to store
    :param name: Name of stored file
    """
    with open('./raw/' + name + '.pickle', 'wb') as f:
        return pickle.dump(file, f)


def open_from_raw(name):
    """
    Open a file with given name from the raw subdirectory.

    :param name: File name
    :return: Fetched file
    """
    with open('./raw/' + name + '.pickle', 'rb') as f:
        return pickle.load(f)


if __name__ == '__main__':
    prep("Fetching training data...")
    # Fetch training data
    fetch_train()
    drop()
    
    prep("Fetching test data...")
    # Fetch test data
    fetch_test()
    drop()
