def init_notebook():
    import sys
    sys.path.insert(0, '/nfs/datasets/codesum/rubenNN/')
    
    import os
    os.chdir('../')
    
    # Hide warnings
    import warnings
    warnings.filterwarnings('ignore')
