"""
visualize.py

Visualize processed images.
"""
import argparse
import matplotlib.pyplot as plt

from myutils import *

# Parameters
folder = 'mnist/'
index = 0

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--folder', type=str, default=folder)
    parser.add_argument('--index', type=int, default=index)
    parser.add_argument('--save', type=bool, default=True)
    
    # Parse arguments
    args = parser.parse_args()
    
    # Open file
    img = load_pickle(args.folder + 'processed/images.pickle')[index]
    label = load_pickle(args.folder + 'processed/labels.pickle')[index]
    
    # Plot the file
    save_path = folder + 'images/image{}'.format(args.index) if args.save else None
    fig, ax = plt.subplots(1, 1, figsize=(3, 3))
    create_image(array=img,
                 ax=ax,
                 title='Image {i} - Truth: {t}'.format(i=args.index, t=label))
    if save_path:
        plt.savefig(save_path)
    plt.show()
    plt.close()
