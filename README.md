# Transfer Learning Demo using Keras

## Overview

1. TLDR
2. Project Overview
3. Full walk-through


## 1. TLDR

Get a already trained model and freeze its layers by calling:
```python
for layer in init_model.layers[:-N]:
    layer.trainable = False

x = NextLayer(...)(init_model.output)
out = LastLayer(...)(x)
new_model = Model(inputs=init_model.input, outputs=out)
```
with `N` the number of layers that remain unfrozen from the initial model. 


## 2. Project Overview

* **Configuration** file of the project is `config.py`, this gives an overview of the most prominent parameters of the
project
* **Jupyter Notebook** which gives a detailed and complete walk-through of the full process can be found in the 
'notebooks/' subfolder, a HTML-compiled version of the final notebook can be in here as well (since this is visually a
bit more appealing, and one does not need to install all the dependencies before opening this)
* **MNIST Dataset** can be found in the 'data/mnist/' folder, both the script for fetching the raw images as the 
preprocess of the samples can be found in here
* **TransferLearner** itself can be found in the `transfer_learner.py` file under root



## 3. Full walk-through

### Introduction - Transfer Learning
Transfer Learning provides a mechanism to efficiently use the data you have by incorporating data from a different but
(possibly quite remotely) related problem. The main advantage is that given the fact you obtain only a limited amount of
data to train your models upon, you can 'pre-train' the model on this other related task, and afterwards fine-tune the
model's parameters using the data you have at your disposal.

In this demo, I will manipulate the classic task for the *MNIST database of handwritten digits* which had as initial
goal to recognize handwritten digits. For this manipulation, the model acts as a binary classifier and needs to predict
if a picture of a handwritten image represents a odd number (labelled 1) or not (labelled 0). 

Suppose now the following: 
* We don't have the resources to create our own database of handwritten digits to map to a *odd number identifier*
* It would not be possible to manipulate the MNIST dataset to be suitable for our problem (Note that this is actually 
not the case since it would be very easy to map the MNIST target-labels to a *odd identifier* simply by mapping each 
target over a modulo operation of two, but for the sake of the tutorial we just ignore this)
* Manually curating the MNIST dataset would take to long

In this case, Transfer Learning would be of great use because we can first train our model on recognizing the images and
afterwards specialize the model into recognizing odd numbers.


### That's great and all, but how does it work?

Good question! **First**, we need to create a model to recognize images. Since the input is only black and white, we 
need only one value, lets say a, integer, to represent each pixel value. We make the spectrum of the pixel fixed between 
an interval of 0 and 255 (i.e. 256 different possible values). Note: To make sure that our model generalizes well, we 
will  consider as model inputs the full RGB-spectrum. Transforming black and white images to RGB images is rather easy, 
just duplicate the input value twice! Say for example you had a complete white pixel, which would hold the value 0, now 
this pixel is represented by the value (0,0,0). Same goes for a complete black pixel: 255 becomes (255,255,255). A grey 
value in between, say for example 142, would become (142,142,142). I think you start to see the pattern.

**Next**, now we have defined the format our processed images must be in, which is also the input dimension of our 
model. We can now start to build the model itself. Since we are trying to recognize images, convolutional neural 
networks would  be a great fit since these are based on the visual cortex found in animals and usually obtain very 
satisfactory results  when applied on a image classification task. The number of convolutional and max-pooling can be 
found empirically, but since this tutorial's main focus is on Transfer Learning, I won't go into detail here. The main 
thing to note, just to avoid any confusion, is that the height and width parameters of the CNNs are unspecified (None). 
This is such that the model generalizes well when new images are applied of different sizes. Note that within one batch 
(e.g. during training) the images must have the same size. The layer where all the magic happens is the 
`GlobalMaxPooling2D` layer, which will transfer the images of a variable size to a vector of fixed size.

After the `GlobalMaxPooling2D` layer (i.e. fixed-size layers) there are two dense layers. The latter dense layer will
be the only layer that changes when going from one problem (i.e. recognizing digits) to the other/our problem 
(recognizing odd numbers). This is because the output of the model changes: classification problem with 10 different
output stages (one for each possible digit), to a binary classification of only 1 single node (True if node is above
a certain threshold, False otherwise).

As hinted before, all the layers except the last output layer will be **frozen** when transferring to the first problem
to the other. This is because at the time of transfer, the model will already have a good knowledge of what digits are
and how to recognize them. The convolutional layers in the network should already understand the key components of a
handwritten digit, and how to extract the most useful information from them.

Note that once the transfer from the first problem to the second problems is executed, it is not possible to train the
model further on the first problem since this would (counter intuitively) decrease the model's performance. The reason
why is that altering only a very small set of the weights in one of the first layers could lead to significant changes
in the lower layers, making everything our last layer has learned worthless since its inputs have changed. Say for 
example that the vector [010001] would result in the index '4' and our last layer (for the second problem) has learned
that this model is not odd. If we would retrain the first layers on the first problems, it could be possible that the
model changes its mind and decides that [010101] would be a better fit to represent '4'. This new input could make
everything the last layer has learned for problem 2 worthless or even incorrect since this input could represent an odd
number in the previous encoding.

 
### Code

Ok, now I've explained the whole rationale in full detail, we're ready to dive into the code. The model itself contains
a **shared network**. This is the network that is trained on the first problem, then frozen and used for our second
problem. *"Frozen"* means that the layer's weights aren't allowed to change anymore. It must be of no surprise that this
methodology only works for similar tasks by now. The code is written using Keras' *Functional Programming* API.

```python
inp = Input(shape=(None, None, 3))

conv11 = Conv2D(filters=16,
                kernel_size=(3, 3),
                activation='relu')(inp)
conv12 = Conv2D(filters=16,
                kernel_size=(3, 3),
                activation='relu')(conv11)
conv13 = Conv2D(filters=16,
                kernel_size=(3, 3),
                activation='relu')(conv12)

maxpool = MaxPooling2D()(conv13)

conv21 = Conv2D(filters=32,
                kernel_size=(3, 3),
                activation='relu')(maxpool)
conv22 = Conv2D(filters=32,
                kernel_size=(3, 3),
                activation='relu')(conv21)
conv23 = Conv2D(filters=32,
                kernel_size=(3, 3),
                activation='relu')(conv22)

# Global MaxPool
global_maxpool = GlobalMaxPooling2D()(conv23)

dense = Dense(128,
              activation='tanh')(global_maxpool)
```

The code used for the **first problem** uses the shared network with an additional final fully connected layer. Since it
is a classification task, a SoftMax will be used as the fully connected layer's activation function. It is then compiled 
using the adam optimizer and sparse categorical crossentropy as its loss function. Note that *sparse* categorical 
crossentropy is exactly the same as the default categorical crossentropy, with the only difference that now the target 
labels are represented by integers instead on one-hot encoded vectors. I.e consider a target vocabulary of size 3 and a 
label '1', the sparse target would then simply be '1' instead of [0,1,0]. Note that 'metrics' aren't used to optimize 
the model, they are only to help visualize the model's training history and can be used to detect overfitted models. 
```python
out = Dense(10,
            activation='softmax',
            name='output')(dense)

self.network = Model(inputs=inp, outputs=out)
self.network.compile(loss='sparse_categorical_crossentropy',
                     optimizer='adam',
                     metrics=['acc'])
```

Next, for our **second problem**, the shared network gets frozen. This can simply be done by toggling the 'trainable'
parameter of the shared network as follows:
```python
shared.trainable = False
```

And our binary classification model will be given a last layer. This last layer has a sigmoid as its activation
function, since it needs to produce a value between 0 (even number) and 1 (odd number). The model will compile using 
binary crossentropy as its loss function and again Adam as its optimizer.
```python
out = Dense(1,
            activation='sigmoid',
            name='output')(dense)

self.network = Model(inputs=inp, outputs=out)
self.network.compile(loss='binary_crossentropy',
                     optimizer='adam',
                     metrics=['acc'])
```