"""
process.py

Rescale the values of the raw data such that each maximum value is 255. Since RGB images are considered, the black-and-
white two-dimensional array is transformed to a three-dimensional array (3x duplicate). Saved shape of each array
representing the image is (28, 28, 3), or in other words (width, height, RGB-index).

Since training-set will be increasing dynamically, test and train sets will be combined in here to represent the full
evaluation set.
"""
import numpy as np

from data.mnist.fetch_data import open_from_raw
from myutils import *


# ---------------------------------------------------> MAIN FILES <--------------------------------------------------- #


def average_images(img_list):
    """
    Normalize the two-dimensional arrays.
    
    :param img_list: List of arrays
    :return: Normalized list of arrays
    """
    img_list_avg = np.zeros(img_list.shape, dtype=float)
    for i, img in enumerate(img_list):
        img_list_avg[i] = img / np.max(img)
    return img_list_avg


def inverse_images(img_list):
    """
    Inverse the values of the normalized arrays.
    
    :param img_list: List of arrays
    :return: Normalized list of arrays
    """
    img_list_inv = np.zeros(img_list.shape, dtype=float)
    for i, img in enumerate(img_list):
        img_list_inv[i] = np.abs(img - 1)
    return img_list_inv


def process_images(img_list):
    """
    Process the images to put them to RGB-format.
    
    :param img_list: List of arrays
    :return: List of arrays
    """
    img_list = average_images(img_list)
    img_list = inverse_images(img_list)
    return transform_rgb(img_list)


def transform_rgb(img_list):
    """
    Transform the normalized list of arrays to RGB-formatted arrays.
    
    :param img_list: Normalized list of arrays
    :return: List of three-dimensional arrays with values between 0 and 255
    """
    samples, width, height = img_list.shape
    img_list_rgb = np.zeros((samples, width, height, 3), dtype=int)
    for i, img in enumerate(img_list):
        x = np.repeat(img, 3).reshape((width, height, 3))
        img_list_rgb[i] = np.round(x * 255)
    return img_list_rgb


# --------------------------------------------------> HELPER FILES <-------------------------------------------------- #


def save_to_processed(file, name, path=''):
    """
    Save a file as a pickle file to the processed subdirectory under the given name.

    :param file: File to store
    :param name: Name of stored file
    :param path: Path to get to current directory
    """
    with open(path + 'processed/' + name + '.json', 'wb') as f:
        return pickle.dump(file, f)


def open_from_processed(name, path=''):
    """
    Open a file with given name from the processed subdirectory.

    :param name: File name
    :param path: Path to get to current directory
    :return: Fetched file
    """
    with open(path + 'processed/' + name + '.pickle', 'rb') as f:
        return pickle.load(f)


if __name__ == '__main__':
    prep("Fetching raw datasets...", key='fetch')
    train_img = open_from_raw('train_images')
    train_lab = open_from_raw('train_labels')
    test_img = open_from_raw('test_images')
    test_lab = open_from_raw('test_labels')
    drop(key='fetch')
    
    # Labels do not change
    prep("Saving labels...", key='process_data')
    full_labels = np.concatenate((train_lab, test_lab))
    store_pickle(full_labels, './processed/labels.pickle')
    drop(key='process_data')
    
    # Process images
    prep("Processing images...", key='process_data')
    train_img = process_images(train_img)
    test_img = process_images(test_img)
    drop(key='process_data')
    
    prep("Saving images...", key='process_data')
    full_images = np.concatenate((train_img, test_img))
    store_pickle(full_images, './processed/images.pickle')
    drop(key='process_data')
    
    print_all_stats()
