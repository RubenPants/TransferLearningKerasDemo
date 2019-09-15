"""
process.py

Rescale the values of the raw data such that each maximum value is 255.
"""
import pickle
import numpy as np

from myutils import *
from data.mnist.fetch_data import open_from_raw


def average_images(img_list):
    """
    Average a list of images as described in the README.
    """
    
    def transform(i):
        img_list_avg[i] = img_list[i]
        img_list_avg[i] = img_list_avg[i] / np.max(img_list_avg[i])  # Normalization
    
    samples, width, height = img_list.shape
    width = width
    height = height
    img_list_avg = np.zeros((samples, width, height), dtype=float)
    for i, img in enumerate(img_list):
        transform(i)
    return img_list_avg


def save_to_processed_1(file, name, path=''):
    """
    Save a file as a pickle file to the processed_1 subdirectory under the given name.

    :param file: File to store
    :param name: Name of stored file
    :param path: Path to get to current directory
    """
    with open(path + 'processed/processed_1/' + name + '.pickle', 'wb') as f:
        return pickle.dump(file, f)


def open_from_processed_1(name, path=''):
    """
    Open a file with given name from the processed_1 subdirectory.

    :param name: File name
    :param path: Path to get to current directory
    :return: Fetched file
    """
    with open(path + 'processed/processed_1/' + name + '.pickle', 'rb') as f:
        return pickle.load(f)


if __name__ == '__main__':
    prep("Fetching raw datasets...")
    train_img = open_from_raw('train_images')
    train_lab = open_from_raw('train_labels')
    test_img = open_from_raw('test_images')
    test_lab = open_from_raw('test_labels')
    drop()
    
    # Labels do not change
    prep("Saving labels...")
    save_to_processed_1(train_lab, 'train_labels')
    save_to_processed_1(test_lab, 'test_labels')
    drop()
    
    # Process images
    prep("Processing images...")
    train_img_avg = average_images(train_img)
    test_img_avg = average_images(test_img)
    drop()
    
    prep("Saving images...")
    save_to_processed_1(train_img_avg, 'train_images')
    save_to_processed_1(test_img_avg, 'test_images')
    drop()
