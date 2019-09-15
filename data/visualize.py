"""
visualize.py

Visualize processed images.
"""
import argparse

from myutils import *

# Parameters
folder = 'mnist/'
index = 0

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--folder', type=str, default=folder)
    parser.add_argument('--index', type=int, default=index)
    parser.add_argument('--save', type=bool, default=False)
    
    # Parse arguments
    args = parser.parse_args()
    
    # Open file
    img = load_pickle(args.folder + 'processed/images.pickle')[index]
    label = load_pickle(args.folder + 'processed/labels.pickle')[index]
    
    # Plot the file
    save_path = folder + 'images/image{}'.format(args.index) if args.save else None
    plot_image(img, save_path=save_path, title='Image {i} - Truth: {t}'.format(i=args.index, t=label))
