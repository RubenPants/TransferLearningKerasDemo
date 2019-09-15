"""
sentiment_learner.py

Model based on convolutional neural networks (CNN) that results to a binary classifier (i.e. True or False).
"""
import config
import numpy as np

from keras.layers import Conv2D, Dense, GlobalMaxPooling2D, Input, MaxPooling2D
from keras.models import Model
from threading import Lock
from myutils import *


class SentimentLearner(object):
    def __init__(self, name, data_path=None):
        """
        Initialisation of static variables.
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
        
        if not self.load_model():
            if not data_path:
                raise Exception("'data_path' must be given when a new model must be created")
            
            # Data
            self.train_images = np.asarray([])
            self.evaluation_images = load_pickle(data_path + 'processed/images.pickle')
            self.train_labels = np.asarray([])
            self.evaluation_labels = load_pickle(data_path + 'processed/labels.pickle')
            
            # Create the network' self
            self.network = self.build_network()
            
            # Save temporal version of new model
            self.save_model()
    
    def __str__(self):
        """
        Create a well suiting name for the model, given its basic parameters.

        :return: String
        """
        return 'sentiment_learner{v}_{n}'.format(v='_v{v:02d}'.format(v=config.VERSION), n=self.name.replace(' ', '_'))
    
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
        
        model = Model(inputs=inp, outputs=out)
        model.compile(loss='binary_crossentropy',
                      optimizer='adam',
                      metrics=['acc'])
        model.summary()
        return model
    
    def train_model(self):
        """
        Train the model for config.EPOCH. Temporary models are saved every config.SAVE_INTERVAL.
        """
        raise NotImplemented
    
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
