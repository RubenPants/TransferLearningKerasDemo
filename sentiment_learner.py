"""
sentiment_learner.py

Model based on convolutional neural networks (CNN) that results to a binary classifier (i.e. True or False).
"""
import collections
import config
import numpy as np

from keras.layers import Conv2D, Dense, GlobalMaxPooling2D, Input, MaxPooling2D
from keras.models import Model
from threading import Lock
from myutils import *


class SentimentLearner(object):
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
        self.train_images = None  # Set of training images
        self.evaluation_images = None  # Set of evaluation images
        self.train_labels = None  # Set of training labels
        self.evaluation_labels = None  # Set of evaluation labels
        
        # Mapping function
        self.mapping = None  # Mapping function from data label to target (Bool)
        
        if not self.load_model():
            if not data_path:
                raise Exception("'data_path' must be given when a new model must be created")
            
            # Data
            self.train_images = None
            self.evaluation_images = load_pickle(data_path + 'processed/images.pickle')
            self.train_labels = None
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
        return 'sentiment_learner_{n}{v}'.format(n=self.name.replace(' ', '_'), v='_v{v:02d}'.format(v=config.VERSION))
    
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
        conv = Conv2D(filters=32,
                      kernel_size=(3, 3),
                      activation='relu',
                      name='conv1_layer1')(inp)
        conv = Conv2D(filters=32,
                      kernel_size=(3, 3),
                      activation='relu',
                      name='conv2_layer1')(conv)
        conv = Conv2D(filters=32,
                      kernel_size=(3, 3),
                      activation='relu',
                      name='conv3_layer1')(conv)
        
        # MaxPool
        maxpool = MaxPooling2D(name='max_pool')(conv)
        
        # Three convolutional layers
        conv = Conv2D(filters=64,
                      kernel_size=(3, 3),
                      activation='relu',
                      name='conv1_layer2')(maxpool)
        conv = Conv2D(filters=64,
                      kernel_size=(3, 3),
                      activation='relu',
                      name='conv2_layer2')(conv)
        conv = Conv2D(filters=64,
                      kernel_size=(3, 3),
                      activation='relu',
                      name='conv3_layer2')(conv)
        
        # Global MaxPool
        global_maxpool = GlobalMaxPooling2D(name='global_max_pool')(conv)
        
        # Fully connected layers
        dense = Dense(256,
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
    
    def eval_and_train(self):
        """
        One step of the process: evaluate to get the most uncertain sample, ask the user to evaluate this model,
        train the model, and save afterwards.
        """
        # Evaluate the model
        index = self.evaluate()
        
        # Ask for user input
        img, label = self.evaluation_images[index], self.evaluation_labels[index]
        plot_image(img, title="Is this image odd?")
        i = input("Is image '{}' odd? [Yes/No]".format(index)).lower()
        
        # Add to training database
        t = 1 if (i in ['yes', 'y', 'true']) else 0 if (i in ['no', 'n', 'false']) else None
        if t is None:
            print("Invalid input!")
        img_add = img.reshape((1,) + img.shape)
        label_add = np.asarray(t).reshape((1,))
        if self.train_images is None:
            self.train_images = img_add
            self.train_labels = label_add
        else:
            self.train_images = np.concatenate((self.train_images, img_add))
            self.train_labels = np.concatenate((self.train_labels, label_add))
        
        # Train the model
        self.train()
    
    def evaluate(self):
        """
        Evaluate all the samples not used for training and plot the distribution. The index of the sample (in evaluation
        set) closest to the model's decision threshold (defined in config) will be returned. The detailed results
        (ground truths) will be stored in the 'evaluation' folder.
        
        :return: Integer: index
        """
        pred = self.network.predict(self.evaluation_images, verbose=1)
        index = np.argmin(abs(pred - 0.5))
        pred_round = pred.round(1)
        
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
            p = 0 if (pred[i][0] < config.THRESHOLD) else 1
            if self.mapping(l) == p:
                c += 1
        plot_bar_graph(counter, title='MNIST - Correct: {}'.format(c / (i + 1)), x_label='Even - Odd')
        
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
        
        :param new_model: SentimentLearner object
        """
        # Model
        self.network = new_model.network  # Container for the model network 'self
        self.current_epoch = new_model.current_epoch  # Amount of epoch the model already trained on
        
        # Data (placeholders)
        self.train_images = new_model.train_images  # Set of training images
        self.evaluation_images = new_model.evaluation_images  # Set of evaluation images
        self.train_labels = new_model.train_labels  # Set of training labels
        self.evaluation_labels = new_model.evaluation_labels  # Set of evaluation labels
        
        # Mapping function
        self.mapping = new_model.mapping  # Mapping function from data label to target (Bool)
