"""
transfer_learner.py

Create a transfer learner by first training a network on the labeled MNIST dataset and then adding an additional layer
to this network to then train on a limited dataset of samples.
"""
import collections
from glob import glob
from random import sample
from threading import Lock
from time import time

import matplotlib.pyplot as plt
import numpy as np
from IPython.display import clear_output
from keras.callbacks import TensorBoard
from keras.layers import Conv2D, Dense, GlobalMaxPooling2D, Input, MaxPooling2D
from keras.models import Model
from keras.optimizers import Adam
from keras_tqdm import TQDMNotebookCallback
from tqdm import tqdm

import config
from myutils import *


class TransferLearner(object):
    def __init__(self, name, data_path=None):
        """
        Initialisation of static variables.
        
        :param name: Name of the model
        :param data_path: Path to (root of) the used data-folder, only used during first initialisation
        """
        # General parameters
        self.name = name
        
        # Model (placeholders)
        self.shared = None  # Container for the shared part of the model's network
        self.network_1 = None  # Container for the network (shared + bottom) on the first problem
        self.network_2 = None  # Container for the network (shared + bottom) on the second problem
        self.current_epoch = 0  # Amount of epoch the model already trained on
        self.is_frozen = False  # Boolean indicating that the shared network's layers are frozen
        
        # System
        self.lock = Lock()  # Lock to dodge concurrency problems
        
        # Data (placeholders)
        self.trained_indices = None  # Set of all indexes already used for training
        self.train_images = None  # List of training images
        self.evaluation_images = None  # List of evaluation images
        self.train_labels = None  # List of training labels
        self.evaluation_labels = None  # List of evaluation labels
        
        if not self.load_model():
            if not data_path:
                raise Exception("'data_path' must be given when a new model must be created")
            
            # Data
            self.trained_indices = set()
            self.evaluation_images = load_pickle(data_path + 'processed/images.pickle')
            self.evaluation_labels = load_pickle(data_path + 'processed/labels.pickle')
            
            # Create the network' self
            self.create_shared_network()
            self.create_model_one()
            
            # Save temporal version of new model
            self.save_model()
    
    def __str__(self):
        """
        Create a well suiting name for the model, given its basic parameters.

        :return: String
        """
        return 'transfer_learner_{n}'.format(n=self.name.replace(' ', '_'))
    
    def __getstate__(self):
        """
        Remove lock from state because it can't be persisted properly.
        """
        state = self.__dict__.copy()
        del state['lock']
        return state
    
    def __setstate__(self, state):
        """
        Add lock back to state when new state fetched.
        """
        self.__dict__.update(state)
        self.lock = Lock()
    
    # -----------------------------------------------> MODEL CREATION <----------------------------------------------- #
    
    def create_model_one(self):
        """
        Compile the first network which purely consists out of all the layers from the shared network, as described in
        the README. Use sparse_categorical_crossentropy since the task at hand is a classification task.
        """
        self.network_1 = Model(inputs=self.shared.input, outputs=self.shared.output)
        self.network_1.compile(loss='sparse_categorical_crossentropy',
                               optimizer='adam',
                               metrics=['acc'])
        self.network_1.summary()
    
    def create_model_two(self):
        """
        Add the bottom layer for the second problem statement (binary classification), as described in the README. The
        shared model's weights will get frozen as well, note that it will not be possible to train the network on the
        first problem anymore after freezing the shared model's layers.
        """
        # Freeze the shared model
        self.is_frozen = True
        self.shared.trainable = False
        for layer in self.shared.layers:
            layer.trainable = False
        
        # Additional layer for the second problem
        out = Dense(1,
                    activation='sigmoid',
                    name='output')(self.shared.output)
        
        self.network_2 = Model(inputs=self.shared.input, outputs=out)
        adam = Adam(lr=config.ADAM_LR)
        self.network_2.compile(loss='binary_crossentropy',
                               optimizer=adam,
                               metrics=['acc'])
        self.network_2.summary()
    
    def create_shared_network(self):
        """
        Create the shared part of the model as described in the README.
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
        
        out = Dense(10,
                    activation='softmax',
                    name='out')(dense)
        
        # Assign the newly created model to the 'shared' parameter
        self.shared = Model(inputs=inp, outputs=out)
    
    # --------------------------------------------> SHARED FUNCTIONALITY <-------------------------------------------- #
    
    def train(self, epochs=1):
        """
        Train the model and save afterwards.
        
        If the shared model is not yet frozen, the network corresponding the first problem statement will be trained.
        Note that it is trained on all data, which isn't how it should be! When done properly, one should ALWAYS
        separate the full dataset in a training and it is a test set. It is also highly encouraged to incorporate a
        validation set as well. But again, to simplify the model in its whole, we will ignore this best practice.
        
        If the shared model is indeed frozen, only manually curated samples will be used to train the model on.
        
        :param epochs: Number of epochs the model will be trained on
        """
        prep('Predicting...', key='training', silent=True)
        
        # Create TensorBoard instantiation
        tensorboard = TensorBoard(log_dir='logs/{}'.format(time()))
        
        # Train the suitable network
        if self.is_frozen:
            self.network_2.fit(
                    x=self.train_images,
                    y=self.train_labels,
                    batch_size=config.BATCH_SIZE,
                    epochs=epochs,
                    verbose=0,
                    callbacks=[TQDMNotebookCallback(leave_inner=True), tensorboard],
            )
            self.current_epoch += epochs  # Epoch count only considered for second network
        else:
            self.network_1.fit(
                    x=self.evaluation_images,
                    y=self.evaluation_labels,
                    batch_size=config.BATCH_SIZE,
                    epochs=epochs,
                    verbose=0,
                    callbacks=[TQDMNotebookCallback(leave_inner=True), tensorboard],
            )
        drop(key='training', silent=True)
        self.save_model()
    
    # -------------------------------------------> NETWORK 1 FUNCTIONALITY <------------------------------------------ #
    
    def evaluate_1(self):
        """
        Evaluate the network corresponding the first problem. This is done by comparing the predictions to the ground
        truth labels provided by the MNIST dataset.
        
        :return (pred, result): A tuple of both the raw predictions as the result of that prediction
        """
        prep('Predicting...', key='evaluation')
        # Get predictions
        preds = self.network_1.predict(self.evaluation_images, verbose=0)
        results = []
        for p in preds:
            results.append(np.argmax(p))
        
        # Evaluate
        c, i = 0, 0
        for r, l in zip(results, self.evaluation_labels):
            if r == l:
                c += 1
        drop(key='evaluation')
        print("Accuracy network 1: ", round(c / (len(results) + 1), 3))
        return preds, results
    
    def visualize_prediction_1(self, index, save=False):
        """
        Visualize the SoftMax activation from the last fully connected layer.
        
        :param index: The index of the evaluation_images that must be evaluated
        :param save: Store the created images
        """
        # Create the prediction
        activation_model = Model(inputs=self.network_1.input, outputs=self.network_1.layers[-1].output)
        img = self.evaluation_images[index]
        img_inp = img.reshape((1,) + img.shape)
        activation = activation_model.predict(img_inp)
        
        # Visualize the input
        print("Input:")
        fig, ax = plt.subplots(1, 1, figsize=(3, 3))
        create_image(array=img,
                     ax=ax,
                     title='Ground Truth: {}'.format(self.evaluation_labels[index]))
        if save:
            plt.savefig('images/input_{e:02d}.png'.format(e=self.current_epoch))
        plt.show()
        plt.close()
        
        # Visualize the activation (SoftMax)
        print("SoftMax activation layer:")
        plt.figure(figsize=(6, 3))
        plt.imshow(activation, vmin=0, vmax=1)
        plt.colorbar(ticks=[0, 1], fraction=0.005)
        plt.xticks(range(10))
        plt.yticks([])
        if save:
            plt.savefig('images/softmax_{e:02d}.png'.format(e=self.current_epoch))
        plt.show()
        plt.close()
    
    # -------------------------------------------> NETWORK 2 FUNCTIONALITY <------------------------------------------ #
    
    def evaluate_2(self):
        """
        Evaluate the network corresponding the first problem. This is done by creating a ground-truth mapping between
        the provided MNIST target labels and our problem's target labels (i.e. module 2 of each original label).
        
        :return (pred, result): A tuple of both the raw predictions as the result of that prediction
        """
        prep('Predicting...', key='evaluation')
        # Get predictions
        preds = self.network_2.predict(self.evaluation_images, verbose=0)
        results = []
        for p in preds:
            if p[0] >= config.THRESHOLD:
                results.append(1)
            else:
                results.append(0)
        
        # Evaluate
        c, i = 0, 0
        for r, l in zip(results, self.evaluation_labels):
            if r == (l % 2):  # l % 2 representing the ground truth
                c += 1
        drop(key='evaluation')
        print("Accuracy network 2: ", round(c / (len(results) + 1), 3))
        return preds, results
    
    def curate_batch(self, batch=config.CURATE_BATCH):
        """
        Manually curate 'batch' number of randomly chosen samples within the 'uncertainty' interval, as defined in the
        config file.
        
        :param batch: Number of samples manually curated
        """
        # First evaluate the model to know the distribution of the model
        preds, results = self.evaluate_2()
        
        # Create a set of 'unsure' samples
        unsure = set()
        mn = min(config.THRESHOLD - config.UNCERTAINTY_STEP, self.current_epoch * config.UNCERTAINTY_STEP)
        mx = max(config.THRESHOLD + config.UNCERTAINTY_STEP, 1 - self.current_epoch * config.UNCERTAINTY_STEP)
        for i, p in enumerate(preds):
            if mn < p[0] < mx:
                if i not in self.trained_indices:
                    unsure.add(i)
        
        # Get 'batch' random samples to curate on
        curate_set = sample(unsure, min(batch, len(unsure)))
        
        # Curate the random samples and add to 'trained_indexes'
        for c in tqdm(curate_set):
            img, label = self.evaluation_images[c], self.evaluation_labels[c]
            fig, ax = plt.subplots(1, 1, figsize=(3, 3))
            create_image(array=img, ax=ax, title="Is this image odd? P(odd)={p:.2f}".format(p=round(preds[c][0], 2)))
            
            # Plot the figure
            plt.show()
            
            # Interact with the user to manually curate the sample
            i = input("Is image '{}' odd? [Yes/No] : ".format(c)).lower()
            
            # Clear and close the plot
            clear_output()
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
            
            # Add index to trained_indices
            self.trained_indices.add(c)
    
    def visualize_distribution(self, save=False):
        """
        Visualize the prediction-distribution for the second problem. This is done by first evaluating the model, and
        then displaying the probability-distribution of the last node across all the samples.
        
        :param save: Store the created images
        """
        # Evaluate the model first
        preds, results = self.evaluate_2()
        
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
        for r, l in zip(results, self.evaluation_labels):
            if r == (l % 2):  # l % 2 representing the ground truth
                c += 1
        
        # Plot the distribution
        fig, ax = plt.subplots(1, 1, figsize=(5, 3))
        create_bar_graph(d=counter,
                         ax=ax,
                         title='MNIST - Correct: {}'.format(round(c / (len(results) + 1), 3)),
                         x_label='Even - Odd')
        if save:
            plt.savefig('images/distribution_{e:02d}.png'.format(e=self.current_epoch))
        plt.show()
        plt.close()
    
    def visualize_prediction_2(self, index, save=False):
        """
        Visualize the last two layers of the second network and their connections, these are:
         * The SoftMax activation from the second last fully connected layer (fixed)
         * The weights between the second last and the last layer (excluding the bias weight)
         * The sigmoid activation
        
        :param index: The index of the evaluation_images that must be evaluated
        :param save: Store the created images
        """
        # Create the prediction
        outputs = [layer.output for layer in self.network_2.layers[-2:]]
        activation_model = Model(inputs=self.network_2.input, outputs=outputs)
        img = self.evaluation_images[index]
        img_inp = img.reshape((1,) + img.shape)
        activation = activation_model.predict(img_inp)
        
        # Visualize the input
        print("Input:")
        fig, ax = plt.subplots(1, 1, figsize=(3, 3))
        create_image(array=img,
                     ax=ax,
                     title='Ground Truth: {}'.format(self.evaluation_labels[index]))
        plt.show()
        plt.close()
        
        # Visualize the activation (SoftMax)
        print("SoftMax activation layer:")
        plt.figure(figsize=(6, 3))
        plt.imshow(activation[0], vmin=0, vmax=1)
        plt.colorbar(ticks=[0, 1], fraction=0.005)
        plt.xticks(range(10))
        plt.yticks([])
        plt.show()
        plt.close()
        
        # Connection weights
        print("Connection weights between last two layers:")
        plt.figure(figsize=(6, 3))
        plt.imshow(np.asarray([[x[0] for x in self.network_2.layers[-1].get_weights()[0]]]), vmin=-1, vmax=1)
        plt.colorbar(ticks=[-1, 0, 1], fraction=0.005)
        plt.xticks(range(10))
        plt.yticks([])
        if save:
            plt.savefig('images/weights_{e:02d}.png'.format(e=self.current_epoch))
        plt.show()
        plt.close()
        
        # Sigmoid activation
        print("Sigmoid activation: value={v:.2f}".format(v=round(activation[1][0][0], 2)))
        plt.figure(figsize=(1, 1))
        plt.imshow(activation[1], vmin=0, vmax=1)
        plt.colorbar(ticks=[0, 0.5, 1], fraction=0.05)
        plt.xticks([])
        plt.yticks([])
        if save:
            plt.savefig('images/sigmoid_{e:02d}.png'.format(e=self.current_epoch))
        plt.show()
        plt.close()
    
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
                models = glob(mdl_path + '*.pickle')
                if len(models) == 0:
                    raise OSError("No models found")
                
                # Give path to load model from
                full_path = models[0]
            
            model = load_pickle(full_path)
            self.update_model(model)
            print("Model '{m}' loaded successfully! Current epoch: {e:d}".format(m=str(self), e=self.current_epoch))
            
            # Show summary of the model
            if self.is_frozen:
                print("Current model: Model for problem 2")
                self.network_2.summary()
            else:
                print("Current model: Model for problem 1")
                self.network_1.summary()
            
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
    
    def print_stats(self):
        """
        Print out the most important parameters of the network.
        """
        items = {
            'current_epoch': self.current_epoch,
            'is_frozen':     self.is_frozen,
            'network_1':     self.network_1,
            'network_2':     self.network_2,
            'shared':        self.shared,
        }
        print(get_fancy_string_dict(items, '{} - Parameters'.format(str(self))))
    
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
        self.shared = new_model.shared  # Container for the shared part of the model's network
        self.network_1 = new_model.network_1  # Container for the network (shared + bottom) on the first problem
        self.network_2 = new_model.network_2  # Container for the network (shared + bottom) on the second problem
        self.current_epoch = new_model.current_epoch  # Amount of epoch the model already trained on
        self.is_frozen = new_model.is_frozen  # Boolean indicating that the shared network's layers are frozen
        
        # Data (placeholders)
        self.trained_indices = new_model.trained_indices  # Set of indexes already used for training
        self.train_images = new_model.train_images  # List of training images
        self.evaluation_images = new_model.evaluation_images  # List of evaluation images
        self.train_labels = new_model.train_labels  # List of training labels
        self.evaluation_labels = new_model.evaluation_labels  # List of evaluation labels
