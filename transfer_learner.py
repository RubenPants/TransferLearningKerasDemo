"""
transfer_learner.py

Model based on convolutional neural networks (CNN) that results to a binary classifier (i.e. True or False).

TODO: Transfer learn the model first
TODO: Init phase (INIT_BATCH) on random samples instead of most unsure?
"""
import collections
from threading import Lock

import matplotlib.pyplot as plt
import numpy as np
from IPython.display import clear_output
from keras.layers import Conv2D, Dense, GlobalMaxPooling2D, Input, MaxPooling2D
from keras.models import Model

import config
from myutils import *


class TransferLearner(object):
    def __init__(self, name, data_path=None, mapping=None):
        """
        Initialisation of static variables.
        
        :param name: Name of the model
        :param data_path: Path to (root of) the used data-folder, only used during first initialisation
        :param mapping: Mapping function between the ground-truth labels and the target-labelling (0 or 1)
        """
        # General parameters
        self.name = name
        
        # Model (placeholders)
        self.network = None  # Container for the model network 'self
        self.current_epoch = 0  # Amount of epoch the model already trained on
        
        # System
        self.lock = Lock()  # Lock to dodge concurrency problems
        
        # Data (placeholders)
        self.trained_indexes = None  # Set of all indexes already used for training
        self.train_images = None  # List of training images
        self.evaluation_images = None  # List of evaluation images
        self.train_labels = None  # List of training labels
        self.evaluation_labels = None  # List of evaluation labels
        self.pred = None  # List of predictions corresponding evaluation_images
        
        # Mapping function
        self.mapping = None  # Mapping function from data label to target (Bool)
        
        if not self.load_model():
            if not data_path:
                raise Exception("'data_path' must be given when a new model must be created")
            
            # Data
            self.trained_indexes = set()
            self.evaluation_images = load_pickle(data_path + 'processed/images.pickle')
            self.evaluation_labels = load_pickle(data_path + 'processed/labels.pickle')
            
            # Mapping function
            self.mapping = mapping
            
            # Create the network' self
            self.build_network()
            
            # Save temporal version of new model
            self.save_model()
    
    def __str__(self):
        """
        Create a well suiting name for the model, given its basic parameters.

        :return: String
        """
        return 'transfer_learner_{n}{v}'.format(n=self.name.replace(' ', '_'), v='_v{v:02d}'.format(v=config.VERSION))
    
    def __getstate__(self):
        state = self.__dict__.copy()
        del state['lock']
        return state
    
    def __setstate__(self, state):
        self.__dict__.update(state)
        self.lock = Lock()
    
    # ------------------------------------------------> MAIN METHODS <------------------------------------------------ #
    
    def build_network(self):
        """
        Create and compile the model, it is recommended to print out the summary as well.
        """
        # RGB input of non-defined size
        inp = Input(shape=(None, None, 3),  # Excluding batch-size
                    name='input')
        
        # Three convolutional layers
        conv11 = Conv2D(filters=16,
                        kernel_size=(3, 3),
                        activation='relu',
                        name='conv1_layer1')(inp)
        conv12 = Conv2D(filters=16,
                        kernel_size=(3, 3),
                        activation='relu',
                        name='conv2_layer1')(conv11)
        conv13 = Conv2D(filters=16,
                        kernel_size=(3, 3),
                        activation='relu',
                        name='conv3_layer1')(conv12)
        
        # MaxPool
        maxpool = MaxPooling2D(name='max_pool')(conv13)
        
        # Three convolutional layers
        conv21 = Conv2D(filters=32,
                        kernel_size=(3, 3),
                        activation='relu',
                        name='conv1_layer2')(maxpool)
        conv22 = Conv2D(filters=32,
                        kernel_size=(3, 3),
                        activation='relu',
                        name='conv2_layer2')(conv21)
        conv23 = Conv2D(filters=32,
                        kernel_size=(3, 3),
                        activation='relu',
                        name='conv3_layer2')(conv22)
        
        # Global MaxPool
        global_maxpool = GlobalMaxPooling2D(name='global_max_pool')(conv23)
        
        # Fully connected layers
        dense = Dense(128,
                      activation='tanh',
                      name='dense')(global_maxpool)
        
        out = Dense(1,
                    activation='sigmoid',
                    name='output')(dense)
        
        self.network = Model(inputs=inp, outputs=out)
        self.network.compile(loss='binary_crossentropy',
                             optimizer='adam',
                             metrics=['acc'])
        self.network.summary()
    
    def iterate(self, train=True):
        """
        One step of the process: evaluate to get the most uncertain sample, ask the user to evaluate this model,
        train the model, and save afterwards.
        
        :param train: Train the network after evaluation
        """
        # Create plot placeholder
        # plt.figure()
        # grid = plt.GridSpec(1, 3, wspace=0.4, hspace=0.3)
        # ax1 = plt.subplot(grid[0, :2])
        # ax2 = plt.subplot(grid[0, 2])
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 3))
        
        # Evaluate the model
        index = self.evaluate(eval_new=train, ax=ax1)
        
        # Ask for user input
        img, label = self.evaluation_images[index], self.evaluation_labels[index]
        create_image(array=img, ax=ax2, title="Is this image odd?")
        
        # Plot the figure
        clear_output()
        plt.show()
        
        # Interact with the user
        i = input("Is image '{}' odd? [Yes/No] : ".format(index)).lower()
        
        # Close the plot
        plt.close()
        
        # Add to training database
        t = 1 if (i in ['yes', 'y', 'true']) else 0 if (i in ['no', 'n', 'false']) else None
        if t is None:
            raise Exception("Invalid input!")
        img_add = img.reshape((1,) + img.shape)
        label_add = np.asarray(t).reshape((1,))
        if self.train_images is None:
            self.train_images = img_add
            self.train_labels = label_add
        else:
            self.train_images = np.concatenate((self.train_images, img_add))
            self.train_labels = np.concatenate((self.train_labels, label_add))
        
        if train:
            # Train the model
            self.train()
    
    def evaluate(self, ax, eval_new):
        """
        Evaluate all the samples not used for training and plot the distribution. The index of the sample (in evaluation
        set) closest to the model's decision threshold (defined in config) will be returned. The detailed results
        (ground truths) will be stored in the 'evaluation' folder.
        
        :param ax: Axis on which the figure will be plotted
        :param eval_new: Evaluate all the samples in evaluation_images again
        :return: Integer: index
        """
        if eval_new or self.pred is None:
            self.pred = self.network.predict(self.evaluation_images, verbose=1)
        pred_index = []  # List containing tuples of (index, value)
        for i, p in enumerate(self.pred):
            pred_index.append((i, abs(p - config.THRESHOLD)))
        pred_index = sorted(pred_index, key=lambda pi: pi[1])
        x = 0
        while pred_index[x][0] in self.trained_indexes:
            x += 1
        self.trained_indexes.add(pred_index[x][0])
        index = pred_index[x][0]
        pred_round = self.pred.round(1)
        
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
        for i, l in enumerate(self.evaluation_labels):
            p = 0 if (self.pred[i][0] < config.THRESHOLD) else 1
            if self.mapping(l) == p:
                c += 1
        create_bar_graph(d=counter,
                         ax=ax,
                         title='MNIST - Correct: {}'.format(round(c / (i + 1), 3)),
                         x_label='Even - Odd')
        
        return index
    
    def train(self):
        """
        Train the model and save afterwards.
        """
        self.network.fit(
                x=self.train_images,
                y=self.train_labels,
                batch_size=config.BATCH_SIZE,
        )
        self.save_model()
    
    # -----------------------------------------------> HELPER METHODS <----------------------------------------------- #
    
    def load_model(self, full_path=None):
        """
        Load a previously trained model and print out its summary.

        :param full_path: [Optional] Path and name (including .pickle) to directory where model is saved

        :return: True: Model loaded and parameters updated successfully | False: Model loading failed
        """
        try:
            self.lock.acquire()
            prep("Loading the model...", key='mdl_load_save')
            
            # No concrete model requested, fetch most-suiting model (based on epoch and model-age)
            if not full_path:
                # Create model name
                mdl_path = "models/{m}".format(m=str(self))
                
                # Search for models that satisfy parameters
                models = glob.glob(mdl_path + '*.pickle')
                if len(models) == 0:
                    raise OSError("No models found")
                
                # Give path to load model from
                full_path = models[0]
            
            model = load_pickle(full_path)
            self.update_model(model)
            print("Model '{m}' loaded successfully! Current epoch: {e:d}".format(m=str(self), e=self.current_epoch))
            
            # Show summary of the model
            self.network.summary()
            
            # Flag that loading was successful
            return True
        except OSError:  # No models found
            print("No model found")
            
            # Create subfolders if don't yet exist
            get_subfolder(path='./', subfolder='models/')
            
            # Flag that loading failed
            return False
        finally:
            drop(key='mdl_load_save')
            self.lock.release()
    
    def save_model(self, full_path=None):
        """
        Save the current model.

        :param full_path: [optional] Path and name (including .h5) to directory where model must be saved
        """
        try:
            self.lock.acquire()
            prep("Saving the model...", key='mdl_load_save')
            
            path = full_path if full_path else 'models/{n}.pickle'.format(n=str(self))
            store_pickle(self, path)
        finally:
            drop(key='mdl_load_save')
            self.lock.release()
    
    def update_model(self, new_model):
        """
        Update the current model's parameters with the new model's parameters.
        
        :param new_model: TransferLearner object
        """
        # Model
        self.network = new_model.network  # Container for the model network 'self
        self.current_epoch = new_model.current_epoch  # Amount of epoch the model already trained on
        
        # Data (placeholders)
        self.trained_indexes = new_model.trained_indexes  # Set of indexes already used for training
        self.train_images = new_model.train_images  # List of training images
        self.evaluation_images = new_model.evaluation_images  # List of evaluation images
        self.train_labels = new_model.train_labels  # List of training labels
        self.evaluation_labels = new_model.evaluation_labels  # List of evaluation labels
        self.pred = new_model.pred  # List of predictions corresponding evaluation_images
        
        # Mapping function
        self.mapping = new_model.mapping  # Mapping function from data label to target (Bool)