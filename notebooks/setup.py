"""
setup.py

Setup the notebooks. This script will put the current working directory back to project root and disables warning in
the notebooks (since these are not so beautiful to look at).
"""


def init_notebook():
    import sys
    sys.path.insert(0, '/nfs/datasets/codesum/rubenNN/')
    
    import os
    os.chdir('../')
    
    # Hide warnings
    import warnings
    warnings.filterwarnings('ignore')
