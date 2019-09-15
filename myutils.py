"""
myutils.py

Share utils used across the project.

Utils overview:
 * CSV: transform_csv
 * DICT: append_dict, load_dict, store_dict
 * JSON: append_json, load_json, store_json
 * PLOT: create_bar_plot
 * SYSTEM: create_subfolder, python2_to_3
 * TIMING: drop, prep, status_out, total_time
"""
import ast
import doctest
import glob
import json
import matplotlib.pyplot as plt
import os
import subprocess
import sys

from matplotlib.ticker import MaxNLocator
from timeit import default_timer as timer


# -------------------------------------------------------> CSV <------------------------------------------------------ #

def load_csv(full_path):
    """
    Load the CSV-file stored under 'full_path', and return it as a list of lists (rows).
    
    :param full_path: Path with name of CSV file (including '.csv')
    :return: List or FileNotFound (Exception)
    """
    result = []
    for line in open(full_path, 'r', encoding="utf8"):
        result.append(line.split('\t'))
    return result


# ------------------------------------------------------> DICT <------------------------------------------------------ #


def append_dict(full_path, new_dict):
    """
    Append existing dictionary if exists, otherwise create new.
    
    :param full_path: Path with name of JSON file (including '.json')
    :param new_dict: The JSON file that must be stored
    """
    files = glob.glob(full_path)
    
    if files:
        # Append new json
        with open(full_path, 'r') as f:
            original = json.load(f)
        
        for k in new_dict:
            original[k] += new_dict[k]
        
        store_json(new_json=original,
                   full_path=full_path,
                   indent=2)
    else:
        # Create new file to save json in
        store_json(new_json=new_dict,
                   full_path=full_path,
                   indent=2)


def clip_dict(full_path, i):
    """
    Clip the dictionary to only the first i elements, if the type of a key's value is a list.
    
    :param full_path: Path with name of JSON file (including '.json')
    :param i: Clipping index: only the first i elements in a list are considered
    """
    with open(full_path, 'r') as f:
        original = json.load(f)
    for k in original:
        v = original[k]
        if type(v) == list:
            original[k] = v[:i]
    store_json(new_json=original,
               full_path=full_path,
               indent=2)


def get_fancy_string_dict(d, title=None):
    """
    Return the string result of the fancy-print for a given dictionary.
    
    :param d: Dictionary
    :param title: [Optional] Title before print
    :return: String
    """
    s = title + '\n' if title else ''
    space = max(map(lambda x: len(x), d.keys()))
    for k, v in d.items():
        s += '\t{key:>{s}s} : {value}\n'.format(s=space, key=k, value=v)
    return s


def load_dict(full_path):
    """
    Load the dictionary stored under 'full_path'.
    
    :param full_path: Path with name of JSON file (including '.json')
    :return: Dictionary or FileNotFound (Exception)
    """
    return load_json(full_path)


def update_dict(full_path, new_dict):
    """
    Update existing dictionary if exists, otherwise create new.
    
    :param full_path: Path with name of JSON file (including '.json')
    :param new_dict: The JSON file that must be stored
    """
    files = glob.glob(full_path)
    
    if files:
        # Append new json
        with open(full_path, 'r') as f:
            original = json.load(f)
        
        original.update(new_dict)
        
        store_json(new_json=original,
                   full_path=full_path,
                   indent=2)
    else:
        # Create new file to save json in
        store_json(new_json=new_dict,
                   full_path=full_path,
                   indent=2)


# ------------------------------------------------------> JSON <------------------------------------------------------ #


def append_json(full_path, new_json):
    """
    Append existing JSON file if exists, otherwise create new json.
    
    :param full_path: Path with name of JSON file (including '.json')
    :param new_json: The JSON file that must be stored
    """
    files = glob.glob(full_path)
    
    if files:
        # Append new json
        with open(full_path, 'r') as f:
            original = json.load(f)
        
        original += new_json
        
        store_json(new_json=original,
                   full_path=full_path,
                   indent=2)
    else:
        # Create new file to save json in
        store_json(new_json=new_json,
                   full_path=full_path,
                   indent=2)


def load_json(full_path):
    """
    Load the JSON file stored under 'full_path'.
    
    :param full_path: Path with name of JSON file (including '.json')
    :return: JSON or FileNotFound (Exception)
    """
    with open(full_path, 'r') as f:
        return json.load(f)


def store_json(new_json, full_path, indent=2):
    """
    Write a JSON file, if one already exists under the same 'full_path' then this file will be overwritten.
    
    :param full_path: Path with name of JSON file (including '.json')
    :param new_json: The JSON file that must be stored
    :param indent: Indentation used to pretty-print JSON
    """
    with open(full_path, 'w') as f:
        json.dump(new_json, f, indent=indent)


# ------------------------------------------------------> PLOT <------------------------------------------------------ #

def create_bar_plot(d, prune=0.95, save_path=None, sort_keys=True, title='', x_label=None, y_label=None):
    """
    Create a bar-plot for the given dictionary.
    
    :param d: Dictionary that must be plotted
    :param prune: Display only the first _ percentage of the plot
    :param save_path: None: do not save | String: save the plot under the given path
    :param sort_keys: Sort based on key-value
    :param title: Title of the plot
    :param x_label: Label of the x-axis
    :param y_label: label of the y-axis
    """
    prep("Plotting", key='creating_plot', silent=True)
    # Abstract keys and values from dictionary
    keys, values = extend_and_split_dictionary(d, sort_keys=sort_keys)
    
    # Sort on value (increasing)
    keys, values = zip(*sorted(zip(keys, values)))
    
    # Maximum values on the respective axis
    x_max = len(values) - 1  # Since key 0 was also taken in consideration
    y_max = max(values)
    
    # Only visualize first <prune> percent of samples (exclude outliers)
    if type(values[0]) == int:
        total = sum(values)
        keep = round(total * prune)
        index = len([i for i in range(x_max + 1) if sum(values[:i]) <= keep])
        if index == 1:
            index += 1
        values = values[:index]
        keys = keys[:index]
        if (x_max + 1) != len(values):
            title += ' - first ' + str(round(prune * 100)) + '%'
        y_label = y_label + ' - max: ' + str(y_max) if y_label else 'max: ' + str(y_max)
    if not keys or type(keys[0]) == int:
        x_label = x_label + ' - max: ' + str(x_max) if x_label else 'max: ' + str(x_max)
    
    ax = plt.figure(figsize=(8, 8)).gca()
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    width = 1 if len(keys) > 50 else 0.8  # 0.8 is default
    plt.bar(range(len(values)), list(values), width=width)
    if keys:
        key_length = len(keys)
        step = key_length // 50 + 1  # Max 50 labels on x-axis
        plt.xticks(range(0, key_length, step), list(keys[0::step]), rotation='vertical')
    else:
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.show()
    plt.close()
    drop(key='creating_plot', silent=True)


def extend_and_split_dictionary(d, sort_keys=True):
    """
    Extend the dictionary with zeros for unused keys. Afterwards, sort the dictionary on increasing key value and split
    the dictionary in a list for keys and a list for values.

    :param d: Dictionary: keys (List), values (List of lists)
    :param sort_keys: Sort on keys if True, otherwise sort on values
    :return: keys (List), values (List)
    """
    value_list = (type(list(d.values())[0]) == list)
    if type(list(d.keys())[0]) == int:
        for i in range(max(list(d.keys()))):
            if i not in d:
                d[i] = [] if value_list else 0
    return zip(*[(key, len(d[key])) for key in sorted(d)]) if sort_keys else \
        zip(*[(key, d[key]) for key in sorted(d, key=d.get)])


# -----------------------------------------------------> SYSTEM <----------------------------------------------------- #

def get_subfolder(path, subfolder, init=True):
    """
    Check if subfolder already exists in given directory, if not, create one.
    
    :param path: Path in which subfolder should be located (String)
    :param subfolder: Name of the subfolder that must be created (String)
    :param init: Initialize folder with __init__.py file (Bool)
    :return: Path name if exists or possible to create, raise exception otherwise
    """
    if subfolder and subfolder[-1] not in ['/', '\\']:
        subfolder += '/'
    
    # Path exists
    if os.path.isdir(path):
        if not os.path.isdir(path + subfolder):
            # Folder does not exist, create new one
            os.mkdir(path + subfolder)
            if init:
                with open(path + subfolder + '__init__.py', 'w') as f:
                    f.write('')
        return path + subfolder
    
    # Given path does not exist, raise Exception
    raise FileNotFoundError("Path '{p}' does not exist".format(p=path))


def python2_to_3(file):
    """
    Transform a Python 2 file to Python 3.
    
    :param file: String
    :return: String or (FileNotFoundError, TypeError, SyntaxError)
    """
    
    def check_code(c):
        """
        Check if code can be parsed and contains code.
        :return: (TypeError, SyntaxError) if not in Python 3 format, ValueError if no code, Nothing otherwise
        """
        tree = ast.parse(c)
        if not tree.body:
            raise ValueError
    
    def python_interpret_to_source(c):
        """
        Transform the code from python interpreter to a source file.
        """
        dt = doctest.DocTestParser()
        samples = dt.get_examples(c)
        s = '\n'.join([s.source for s in samples])
        return s if s else c
    
    try:
        # Check if already in Python 3
        check_code(file)
        return file
    except (TypeError, SyntaxError):
        # Check if interpreter-code (transform if so)
        file = python_interpret_to_source(file)
        
        # Save file as a temporary file
        with open('temp.py', 'w') as f:
            f.write(file)
        
        # Transform 2 -> 3  (Unix only)
        subprocess.call(["2to3", 'temp.py', "-w", "--no-diffs"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        
        # Read transformed Python3 file
        with open('temp.py', 'r') as f:
            file = f.read()
        
        # Check if now correct
        check_code(file)
        return file


# -----------------------------------------------------> TIMING <----------------------------------------------------- #

time_dict = dict()


def drop(key=None, silent=False):
    """
    Stop timing, print out duration since last preparation and append total duration.
    """
    # Update dictionary
    global time_dict
    if key not in time_dict:
        raise Exception("prep() must be summon first")
    time_dict[key]['end'] = timer()
    time_dict[None]['end'] = timer()
    
    # Determine difference
    start = time_dict[key]['start']
    end = time_dict[key]['end']
    diff = end - start
    
    # Fancy print diff
    if not silent:
        status_out(' done, ' + get_fancy_time(diff) + '\n')
    
    # Save total time
    if key is not None:
        if 'sum' not in time_dict[key]:
            time_dict[key]['sum'] = diff
        else:
            time_dict[key]['sum'] += diff


def get_fancy_time(sec):
    """
    Convert a time measured in seconds to a fancy-printed time.
    
    :param sec: Float
    :return: String
    """
    h = int(sec) // 3600
    m = (int(sec) // 60) % 60
    s = sec % 60
    if h > 0:
        return '{h} hours, {m} minutes, and {s} seconds.'.format(h=h, m=m, s=round(s, 2))
    elif m > 0:
        return '{m} minutes, and {s} seconds.'.format(m=m, s=round(s, 2))
    else:
        return '{s} seconds.'.format(s=round(s, 2))


def prep(msg="Start timing...", key=None, silent=False):
    """
    Prepare timing, print out the given message.
    """
    global time_dict
    if key not in time_dict:
        time_dict[key] = dict()
    if not silent:
        status_out(msg)
    time_dict[key]['start'] = timer()
    
    # Also create a None-instance (in case drop() is incorrect)
    if key:
        if None not in time_dict:
            time_dict[None] = dict()
        time_dict[None]['start'] = timer()


def print_all_stats():
    """
    Print out each key and its total (cumulative) time.
    """
    global time_dict
    if time_dict:
        if None in time_dict: del time_dict[None]  # Remove None-instance first
        print("\n\n\n---------> OVERVIEW OF CALCULATION TIME <---------\n")
        keys_space = max(map(lambda x: len(x), time_dict.keys()))
        line = ' {0:^' + str(keys_space) + 's} - {1:^s}'
        line = line.format('Keys', 'Total time')
        print(line)
        print("-" * (len(line) + 3))
        line = '>{0:^' + str(keys_space) + 's} - {1:^s}'
        t = 0
        for k, v in sorted(time_dict.items()):
            t += v['sum']
            print(line.format(k, get_fancy_time(v['sum'])))
        end_line = line.format('Total time', get_fancy_time(t))
        print("-" * (len(end_line)))
        print(end_line)


def status_out(msg):
    """
    Write the given message.
    """
    sys.stdout.write(msg)
    sys.stdout.flush()
