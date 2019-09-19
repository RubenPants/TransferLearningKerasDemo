"""
Functionality used to create GIFs programmatically.
"""
import collections

import matplotlib.pyplot as plt
import numpy as np

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
