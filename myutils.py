"""
myutils.py

Share utils used across the project.

Utils overview:
 * CSV: transform_csv
 * DICT: append_dict, load_dict, store_dict
 * JSON: append_json, load_json, store_json
 * PICKLE: store_pickle, load_pickle
 * PLOT: create_bar_plot
 * SYSTEM: create_subfolder, python2_to_3
 * TIMING: drop, prep, status_out, total_time
"""
import os
import pickle
import sys

from timeit import default_timer as timer


# ------------------------------------------------------> DICT <------------------------------------------------------ #


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


# -----------------------------------------------------> PICKLE <----------------------------------------------------- #

def load_pickle(full_path):
    """
    Load the pickled file stored under 'full_path'.
    
    :param full_path: Path with name of pickle file (including '.pickle')
    :return: Pickled object file or FileNotFound (Exception)
    """
    with open(full_path, 'rb') as f:
        return pickle.load(f)


def store_pickle(file, full_path):
    """
    Pickle and store a file, if one already exists under the same 'full_path' then this file will be overwritten.
    
    :param file: The (object) file that must be pickled
    :param full_path: Path with name of pickled file (including '.pickle')
    """
    with open(full_path, 'wb') as f:
        return pickle.dump(file, f)


# ------------------------------------------------------> PLOT <------------------------------------------------------ #


def create_bar_graph(ax, d, title='', x_label=None):
    """
    Create a bar-plot for the dictionary (counter) d.
    
    :param ax: Axis on which the plot will be plotted
    :param d: Dictionary
    :param title: Title of the plot
    :param x_label: Label of the x-axis
    """
    keys, values = zip(*sorted(d.items()))
    ax.bar(keys, values, width=0.09)
    ax.set_title(title)
    ax.set(xlabel=x_label, ylabel='Number of samples')
    ax.plot()


def create_image(array, ax, title=''):
    """
    Plot the image represented by the array.

    :param array: Three dimensional numpy array
    :param ax: Axis on which the plot will be plotted
    :param title: Title of the figure
    """
    prep("Plot", key='plot', silent=True)
    ax.imshow(array)
    ax.set_title(title)
    ax.plot()


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
