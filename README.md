## Dogs-Vs-Cats CNN
**Goal of this model:**
To build and train a CNN that can accurately identify images of cats from those of Dogs.Moreover, fine-tune a **pretrained VGG16 model** and compare how efficient it is in making predictions from our sequential model that we'll build from scratch.

#### Obtaining The Data

For this model we'll use a random subset of the  Kaggle Dogs Versus Cats competition data set. Check it out at Kaggle website.
**Organizing The Data**
This will involve re-organizing the directory structure to hold the data set. Checkout how I finally organized the data in structure in this repo.


### Prerequisite
Before getting your hands on work consider having this pre-installation on your machine:
* Anaconda (*comes with Jupyter Notebook*)
* Python3.6 or higher version 
* Using Anaconda create an environment for Tensorflow CPU/GPU
* Have a modern web browser, preferably Chrome

**NOTE: Remember to activate your Anaconda environment inside the directory you'll be working from. After activating the environment, use conda to install relevant packages that you'll need.**

## Hands-on-work
#### Image Preparation for a CNN model
I'll walk you through the necessary Image preparation steps that I set up before training this CNN model.

First I made the necessary imports. This include: Numpy, Tensorflow(embedded with Keras API), itertools,shutil, random, glob, matplotlib.pyplot and warnings modules.
Next, I created the `train_path`, `valid_path` and`test_path` variables for which the paths to the train, valid, and test data directories were assigned. 

Using the Keras' `ImageDataGenerator` class I created batches of data from the `train`, `valid` and `test` directories. The `ImageDataGenerator.floe_from_directory()` creates a `DirectoryIterator`, which generates batches of normalized tensor image data from the respective data directories.
#### Visualizing The Training Data
I called the `next(train_batches)` to generate a batch of images and labels from the training set. * **Note:** The size of this batch is determined by the `batch_size` we set when we created `train_batches`.

I then used the plotting function obtained from Tensorflow's documentation to plot the processed images within my Jupyter-Notebook.
Below is a snippet of the code;
![image.png](images/img_process.jpeg)

This what the first processed random batch from the training set looked like. Note that the color appears to be distorted because we applied the VGG16 processing to data sets.It is also important to note that the dogs are represented with the `one-hot encoding` of `[0,1]` and cats represented by `[1,0]`
![image.png](images/img_visual.jpeg)

#### Building and Training The model 
To build the CNN, I used the Keras **Sequential** model.
The model had 6 layers. The first layer of the model was 2-dimensional convolutional layer. It had 32 output filters each with a kernel size of `3x3`, and also used a `relu` activation function. I enabled `zero-padding` by specifying ` padding = 'same'`

- `model = Sequential([  
    Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding = 'same', input_shape=(224,224,3)),
    MaxPool2D(pool_size=(2, 2), strides=2),
    Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding = 'same'),
    MaxPool2D(pool_size=(2, 2), strides=2),
    Flatten(),
    Dense(units=2, activation='softmax')
])`

On the first layer only, I specified the `input_shape`, which is the shape of our data. The images are **224 pixels high** and **224 pixels wide** and have **3 color** channels: RGB. 

`max pooling` layer was used to pool and reduce the data dimensionality

To check out the summary of the model if you have cloned this repo, run this command
- `model.summary()`

#### Compiling The Model
After building the model, I compiled it using the `Adam` optimizer with a learning rate of `0.0001`, a loss of `categorical_cross_entropy` and passed `accuracy` as my performance `metric`.

Below is the code for compiling the model.

- `model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_cross_entropy', metrics=['accuracy'])`

