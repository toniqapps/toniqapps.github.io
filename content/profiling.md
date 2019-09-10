Title: Profiling basic training metrics in Keras
Date: 2019-09-10 09:10
Category: AI
Authors: Rahul Jain

# Profiling basic training metrics in Keras

Performance is critical for machine learning. TensorFlow has a built-in profiler that allows you to record runtime of each ops with very little effort. Then you can visualize the profile result in TensorBoard's Profile Plugin. This tutorial focuses on GPU but the Profile Plugin can also be used with TPUs by following the Cloud TPU Tools.

This tutorial presents very basic examples to help you learn how to enable profiler when developing your Keras model. You will learn how to use the Keras TensorBoard callback to visualize profile result.


## Prerequisites

### Setup

## Network
### Densnet Model model with TensorBoard callback

```
def add_denseblock(input, num_filter = 12, dropout_rate = 0.2):
    temp = input
    for _ in range(l):
        BatchNorm = BatchNormalization()(temp)
        relu = Activation('relu')(BatchNorm)
        Conv2D_3_3 = Conv2D(int(num_filter), (3,3), use_bias=False ,padding='same')(relu)
        if dropout_rate>0:
            Conv2D_3_3 = Dropout(dropout_rate)(Conv2D_3_3)
        concat = Concatenate(axis=-1)([temp, Conv2D_3_3])
        
        temp = concat
        
    return temp
    
def add_transition(input, num_filter = 12, dropout_rate = 0.2):
    global compression
    BatchNorm = BatchNormalization()(input)
    relu = Activation('relu')(BatchNorm)
    Conv2D_BottleNeck = Conv2D(int(num_filter*compression), (1,1), use_bias=False ,padding='same')(relu)
    if dropout_rate>0:
        Conv2D_BottleNeck = Dropout(dropout_rate)(Conv2D_BottleNeck)
    avg = AveragePooling2D(pool_size=(2,2))(Conv2D_BottleNeck)
    
    return avg
    
def output_layer(input):
    BatchNorm = BatchNormalization()(input)
    #relu = Activation('relu')(BatchNorm)
    AvgPooling = AveragePooling2D(pool_size=(2,2))(BatchNorm)
    flat = Flatten()(AvgPooling)
    output = Dense(num_classes, activation='softmax')(flat)
    
    return output
    
    
num_filter = 12
dropout_rate = 0.2
l = 12

input = Input(shape=(32, 32, 3))
First_Conv2D = Conv2D(num_filter, (7,7), use_bias=False ,padding='same')(input)

First_Block = add_denseblock(First_Conv2D, num_filter, dropout_rate)
First_Transition = add_transition(First_Block, num_filter, dropout_rate)

Second_Block = add_denseblock(First_Transition, num_filter, dropout_rate)
Second_Transition = add_transition(Second_Block, num_filter, dropout_rate)

Third_Block = add_denseblock(Second_Transition, num_filter, dropout_rate)
Third_Transition = add_transition(Third_Block, num_filter, dropout_rate)

Last_Block = add_denseblock(Third_Transition,  num_filter, dropout_rate)
output = output_layer(Last_Block)
    
model = Model(inputs=[input], outputs=[output])
model.summary()
```

## Download CIFAR-10 Data

### Create TFRecords for CIFAR-10 Dataset in CWD

```
tfrecords.create("cifar10", "./")
```

### Load TFRecords & Augment the loaded images

~~~~ 
def preprocess(img, img_shape, training):
    img = img * (1. / 255.)
    img_height, img_width, img_depth = img_shape
    if training:
        # Resize the image to add four extra pixels on each side.
        img = tf.image.resize_image_with_crop_or_pad(
            img,
            img_height + 8,
            img_width + 8
        )

        # Randomly crop a [_height, _width] section of the image.
        img = tf.random_crop(img, img_shape)

        # Randomly flip the image horizontally.
        img = tf.image.random_flip_left_right(img)

    #Subtract off the mean and divide by the variance of the pixels.
    img = tf.image.per_image_standardization(img)
    return img


train_dataset = tfrecords.load("cifar10", ["./train.tfrecords"], batch_size, preprocess, training=True)
test_dataset = tfrecords.load("cifar10", ["./eval.tfrecords"], batch_size, preprocess, training=False)
~~~~ 

When creating TensorBoard callback, you can specify the batch num you want to profile. By default, TensorFlow will profile the second batch, because many one time graph optimizations run on the first batch. You can modify it by setting profile_batch. You can also turn off profiling by setting it to 0.

This time, you will profile on the third batch.
~~~~
log_dir="logs/profile/" + datetime.now().strftime("%Y%m%d-%H%M%S")

tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1, profile_batch = 3)
~~~~


## Train Model
~~~~
model.fit(
    train_dataset,
    epochs=5,
    steps_per_epoch=20,
    callbacks=[tensorboard_callback],
)
~~~~

![alt text](/images/model_fit_result.png)

## Visualizing profile result using TensorBoard

Unfortunately, due to #1913, you cannot use TensorBoard in Colab to visualize profile result. You are going to download the logdir and start TensorBoard on your local machine.

Compress logdir:

~~~~
!tar -zcvf logs.tar.gz logs/profile/
~~~~

Download logdir.tar.gz by right-clicking it in “Files” tab.

<img src="/images/file_download.png" width="200" height="200" />

Please make sure you have the latest TensorBoard installed on you local machine as well. 

[Setup TensorBoard on MacOS](http://toniqapps.github.io/tensorboard.md)

Execute following commands on your local machine:

```
cd download/directory
$ tar -zxvf logs.tar.gz
$ tensorboard --logdir=logs/ --port=6006
```

Open a new tab in your Chrome browser and navigate to localhost:6006 and then click “Profile” tab. You may see the profile result like this:

<img src="/images/tensorboard.png" width="1000"/>

