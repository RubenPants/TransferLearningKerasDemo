"""
Functionality used to create GIFs programmatically.
"""
import collections
from glob import glob

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from myutils import create_bar_graph


def create_image_model_state(model):
    """
    Evaluate the model at its current state and summarize this evaluation in a single picture.
    
    :param model: TransferLearner
    """
    # --> Calculate distribution <-- #
    # Evaluate the model first
    preds, results = model.evaluate_2()
    
    # Round the predictions to one floating digit
    pred_round = preds.round(1)
    
    # Init counter
    counter = collections.Counter()
    for i in range(11):
        counter[i / 10] = 0
    
    # Count distribution
    for p in pred_round:
        counter[p[0]] += 1
    counter = dict(sorted(counter.items()))
    
    # Evaluate to ground truth
    c, i = 0, 0
    for r, l in zip(results, model.evaluation_labels):
        if r == (l % 2):  # l % 2 representing the ground truth
            c += 1
    
    # --> Create plot <-- #
    fig = plt.figure(constrained_layout=True)
    gs = fig.add_gridspec(6, 5)
    plt.suptitle("Number of curated samples: {e:02d}".format(e=model.current_epoch * 4))
    
    distribution = fig.add_subplot(gs[:4, :])
    create_bar_graph(d=counter,
                     ax=distribution,
                     title='MNIST - Correct: {}'.format(round(c / (len(results) + 1), 3)),
                     x_label='Even - Odd')
    
    weights = fig.add_subplot(gs[4:, :])
    plt.imshow(np.asarray([[x[0] for x in model.network_2.layers[-1].get_weights()[0]]]), vmin=-1, vmax=1)
    plt.colorbar(ticks=[-1, 0, 1], fraction=0.005)
    plt.xticks(range(10))
    plt.yticks([])
    weights.set_title('connection weights')
    
    plt.savefig('images/gif_{e:02d}'.format(e=model.current_epoch))
    plt.show()
    plt.close()


def create_gif(rel_path='images/'):
    """
    Create a gif using all the saved 'gif_' png files.
    """
    images = glob('{p}gif_*.png'.format(p=rel_path))
    img, images = images[0], images[1:]
    im_start = Image.open(img)
    img_list = [im_start for _ in range(4)]  # 5x the same image in the beginning
    img_list += [Image.open(i) for i in images]
    im_end = Image.open(images[-1])
    img_list += [im_end for _ in range(4)]  # 5x the same image in the end
    im_start.save('{p}final_fig.gif'.format(p=rel_path),
                  save_all=True,
                  append_images=img_list,
                  duration=1000,
                  loop=0)


if __name__ == '__main__':
    create_gif(rel_path='')
