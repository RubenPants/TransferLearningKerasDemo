# Transfer Learning Demo using Keras

## Overview

1. **TLDR** - For the ones in a hurry
2. **Project Overview** - For the real hackers who want to dive straight into the code
3. **Full walk-through** - For those who want to learn a bit and have 15 minutes of spare time

<p align="center">
  <img src="https://github.com/RubenPants/TransferLearningKerasDemo/blob/master/images/final_fig.gif"/>
</p>


## 1. TLDR

Get an already trained model and freeze its layers by calling:
```python
for layer in init_model.layers[:-N]:
    layer.trainable = False

x = NextLayer(...)(init_model.output)
out = LastLayer(...)(x)
new_model = Model(inputs=init_model.input, outputs=out)
```
with `N` the number of layers that remain unfrozen from the initial model. 



## 2. Project Overview

* **Configuration** file of the project is `config.py`, this gives an overview of the most prominent parameters of the project
* **Jupyter Notebook** which gives a detailed and complete walk-through of the full process can be found in the 'notebooks/' subfolder, a HTML-compiled version of the final notebook can be in here as well (since this is visually a bit more appealing, and one does not need to install all the dependencies before opening this)
* **MNIST Dataset** can be found in the 'data/mnist/' folder, both the script for fetching the raw images as the preprocess of the samples can be found in here
* **Transfer Learner** itself can be found in the `transfer_learner.py` file under root



## 3. Full walk-through

### Introduction - Transfer Learning
Transfer Learning provides a mechanism to efficiently use the data you have by incorporating data from a different but (possibly quite remotely) related problem. The main advantage is that given the fact you obtain only a limited amount of data to train your models upon, you can 'pre-train' the model on this other related task, and afterwards fine-tune the model's parameters using the data you have at your disposal.

In this demo, I will manipulate the classic task for the *MNIST database of handwritten digits* which had as initial goal to recognize handwritten digits. For this manipulation, the model acts as a binary classifier and needs to predict if a picture of a handwritten image represents a odd number (labelled 1) or not (labelled 0). 

Suppose now the following: 
* We don't have the resources to create our own database of handwritten digits to map to a *odd number identifier*
* It would not be possible to manipulate the MNIST dataset to be suitable for our problem (Note that this is actually not the case since it would be very easy to map the MNIST target-labels to a *odd identifier* simply by mapping each target over a modulo operation of two, but for the sake of the tutorial we just ignore this)
* Manually curating the MNIST dataset would take to long

In this case, Transfer Learning would be of great use because we can first train our model on recognizing the images and afterwards specialize the model into recognizing odd numbers.


### That's great and all, but how does it work?

Good question! **First**, we need to create a model to recognize images. Since the input is only black and white, we need only one value, lets say a, integer, to represent each pixel value. We make the spectrum of the pixel fixed between an interval of 0 and 255 (i.e. 256 different possible values). Note: To make sure that our model generalizes well, we will  consider as model inputs the full RGB-spectrum. Transforming black and white images to RGB images is rather easy, just duplicate the input value twice! Say for example you had a complete white pixel, which would hold the value 0, now this pixel is represented by the value (0,0,0). Same goes for a complete black pixel: 255 becomes (255,255,255). A grey value in between, say for example 142, would become (142,142,142). I think you start to see the pattern.

**Next**, now we have defined the format our processed images must be in, which is also the input dimension of our model. We can now start to build the model itself. Since we are trying to recognize images, convolutional neural networks would  be a great fit since these are based on the visual cortex found in animals and usually obtain very satisfactory results  when applied on a image classification task. The number of convolutional and max-pooling can be found empirically, but since this tutorial's main focus is on Transfer Learning, I won't go into detail here. The main thing to note, just to avoid any confusion, is that the height and width parameters of the CNNs are unspecified (None). This is such that the model generalizes well when new images are applied of different sizes. Note that within one batch (e.g. during training) the images must have the same size. The layer where all the magic happens is the `GlobalMaxPooling2D` layer, which will transfer the images of a variable size to a vector of fixed size.

After the `GlobalMaxPooling2D` layer (i.e. fixed-size layers) there are two dense layers. The latter dense layer will be used for the classification problem. In our second problem, an additional layer will be added upon this 'categorization layer', which will be represent a binary classification layer based on the sigmoid activation function. In other words, the output for the first network (`network_1`) will be of size 10, since there are ten different classes representing each a different index. The output for the second network (`network_2`) on the other hand will have only a single output node that acts as a binary classifier: True if the node's value is above a pre-defined threshold, False otherwise.

As hinted before, all the layers of the first model will be **frozen** when transferring from the first problem to the other. This is because at the time of transfer, the model will already have a good knowledge of what digits are and how to recognize them. The convolutional layers in the network should already understand the key components of a handwritten digit, and how to extract the most useful information from them. Furthermore, it is only a simple mapping to go from digit-categories to the odd-even distribution.

Note that once the transfer from the first problem to the second problems is executed, it is not possible to train the model further on the first problem since this would (counter intuitively) decrease the model's performance. The reason why is that altering only a very small set of the weights in one of the first layers could lead to significant changes in the lower layers, making everything our last layer has learned worthless since its inputs have changed. Say for example that the vector [010001] would result in the index '4' and our last layer (for the second problem) has learned that this model is not odd. If we would retrain the first layers on the first problems, it could be possible that the model changes its mind and decides that [010101] would be a better fit to represent '4'. This new input could make everything the last layer has learned for problem 2 worthless or even incorrect since this input could represent an odd number in the previous encoding.


### Code

Ok, now I've explained the whole rationale in full detail, we're ready to dive into the code. The model itself contains a **shared network**. This is the network that is trained on the first problem, then frozen and used for our second problem. *"Frozen"* means that the layer's weights aren't allowed to change anymore. It must be of no surprise that this methodology only works for similar tasks by now. The code is written using Keras' *Functional Programming* API.

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
global_maxpool = GlobalMaxPooling2D()(conv23)

dense = Dense(128,
              activation='tanh')(global_maxpool)

out = Dense(10,
            activation='softmax')(dense)
```

The code used for the **first problem** only needs to compile the model such that it is suitable for the classification task, which means that a SoftMax will be used as the fully connected layer's activation function. It is then compiled using the Adam optimizer and sparse categorical crossentropy as its loss function. Note that *sparse* categorical crossentropy is exactly the same as the default categorical crossentropy, with the only difference that now the target labels are represented by integers instead on one-hot encoded vectors. I.e consider a target vocabulary of size 3 and a label '1', the sparse target would then simply be '1' instead of [0,1,0]. Note that 'metrics' aren't used to optimize the model, they are only to help visualize the model's training history and can be used to detect overfitted models. 

```python
model = Model(inputs=shared.input, outputs=shared.output)
model.compile(loss='sparse_categorical_crossentropy',
              optimizer='adam',
              metrics=['acc'])
```

Next, for our **second problem**, the shared network gets frozen. To do so, we have to freeze each layer separately (I found out that `shared.trainable = False` doesn't seem to work).
```python
for layer in shared.layers:
    layer.trainable = False
```

And our binary classification model will be given a last layer. This last layer has a sigmoid as its activation
function, since it needs to produce a value between 0 (even number) and 1 (odd number). The model will compile using binary crossentropy as its loss function and again Adam as its optimizer.

```python
out = Dense(1,
            activation='sigmoid')(shared.output)

model = Model(inputs=shared.input, outputs=out)
model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['acc'])
```

### Model Evaluation

In this last section, I will go briefly over the models performance. Let's consider that everything during training on the first problem went follows plan, and we want to start training our second model. In the first stage, the weights connecting the classification layer (CL) and the binary layer (BL) are chosen completely random, which has the following as a result:

<p align="center">
  <img src="https://github.com/RubenPants/TransferLearningKerasDemo/blob/master/images/gif_00.png"/>
</p>

When analyzing the weights between CL and BL, we see that some are better than others. The target-weights we want are high weights ('bright') for the odd numbers, and low ('dark') for the even numbers. Put otherwise, the model has some training to do before it realizes that (e.g.) 9 should get a high weight instead of a low one.

The first 20 manually curated samples, the model asked a lot about the numbers {2, 3, 6}, making that these numbers occurred relatively often in the training set (manually curated samples). This with result that after only five iterations, the model knew very well that {3} is an odd number ('bright' square), and {2, 6} are even numbers ('dark' squares). Note that it mistakenly darkens the square of 1 a little, but this will be fixed afterwards (1 didn't occur in the manually curated set thus far).

<p align="center">
  <img src="https://github.com/RubenPants/TransferLearningKerasDemo/blob/master/images/gif_05.png"/>
</p> 

For the next iterations, the model has a good focus on the numbers {0, 5, 7}. It also improves a little for the numbers {4, 8}.

<p align="center">
  <img src="https://github.com/RubenPants/TransferLearningKerasDemo/blob/master/images/gif_13.png"/>
</p>

The next iterations, the model almost solely asks to curate the number 9, which has a great influence on the weights corresponding this number:

<p align="center">
  <img src="https://github.com/RubenPants/TransferLearningKerasDemo/blob/master/images/gif_20.png"/>
</p>

The only non-optimized number left is number 1, this is because up till now, this number wasn't asked to label by the model, meaning that it wasn't possible for the model to train on this number. Since the model predicts that '1' is an even number with quite some confidence, it takes some time until this group of images representing ones will shift towards the 'odd'-side. Eventually, after only 30 iterations (i.e. 120 manually curated samples), we get the following distribution and CL-BL weights:

<p align="center">
  <img src="https://github.com/RubenPants/TransferLearningKerasDemo/blob/master/images/gif_30.png"/>
</p>

Note that for the last problem where the model needed quite some time to train on number 1, less manually curated 1-samples would result in a comparable result as we obtained here. Once the model has the right curated samples and is trained enough, the weights will eventually update to the right values.

### TensorBoard

In a last update, I've added TensorBoard support during training. TensorBoard is a handy tool to (visually) analyze training progress of your model. This will be done automatically so there is nothing for you to worry about. An example of how to do so in your own project can be found on https://www.youtube.com/watch?v=2U6Jl7oqRkM, it's only a two minute watch! To fire up TensorBoard after training, type `tensorboard --logdir logs/` in a terminal in root.

### Conclusion

In another experiment I did, each number had one 'representative' that was manually curated. After only 8 epochs of training, the model already obtained better results that we got above (with only 10 manually curated samples!). This to show that the choice of (high quality) samples is very important when performing transfer learning. Since the model already has a general understanding of the problem, due to the training on `network_1`, it is better to have less but more qualitative samples, than to have more noisy samples in your dataset.

I hope this tutorial is clear for you, if you find any bugs or incorrect information then please open an issue on the GitHub page and I'll try to look into it as fast as possible!



Have a nice strumming

Ruben
