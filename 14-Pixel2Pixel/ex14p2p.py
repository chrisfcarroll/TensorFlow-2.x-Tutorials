import numpy as np
from numpy import ndarray
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, optimizers, metrics
from tensorflow.python.data import Dataset
from matplotlib import pyplot as plt
import os
import time

assert tf.__version__ >='2'; os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.random.set_seed(3426) ; np.random.seed(3426)
learning_rate=2e-4
beta_1=0.5
batch_size=1
facades_img_width=256
facades_img_height=256


def main():
    generator=Generator()
    generator.build(input_shape=(batch_size,facades_img_width,facades_img_height,3))
    generator.summary()
    discriminator=Discriminator()
    discriminator.build(input_shape=[(batch_size,facades_img_width,facades_img_height,3),(batch_size,facades_img_width,facades_img_height,3)])
    discriminator.summary()
    gen_optimizer=keras.optimizers.Adam(learning_rate=learning_rate, beta_1=beta_1)
    disc_optimizer=keras.optimizers.Adam(learning_rate=learning_rate,beta_1=beta_1)

    train_ds, val_ds = get_facade_dataset() ; print(train_ds.cardinality(),val_ds.cardinality())
    train(train_ds)



def train(train_ds, epochs=100):
    print('Training for {} epochs ...'.format(epochs))
    for epoch in range(1,epochs+1):
        start=time.time()
        for step,inputs in enumerate(train_ds):
            input,target=tf.split(inputs,num_or_size_splits=[3,3],axis=3)
            print('input.shape={}, target.shape={}'.format(input.shape, target.shape))




def generate_and_show_images(model, test_input, target, epoch):
    prediction=model(test_input,training=True) #training=True get us per-batch stats
    plt.figure(figsize=(15,15))
    display_list=[test_input[0], target[0], prediction[0]]
    title=['Input image', 'Ground Truth', 'Predicted Image']
    for i in range(3):
        plt.subplot(1,3, i+1)
        plt.title(title[i])
        plt.imshow(display_list[i] * 0.5 + 0.5) #rescale (-1,1)->(0,1)
        plt.axis('off')
    plt.savefig('images/epoch{:04d}.png'.format(epoch))
    #plt.show()


def discriminator_loss(discr_real,discr_generated):
    # [1,30,30,1] with [1,30,30,1]
    real_loss=keras.losses.binary_crossentropy(
            tf.ones_like(discr_real), discr_real, from_logits=True)
    generated_loss=keras.losses.binary_crossentropy(
            tf.zeros_like(discr_generated), discr_generated, from_logits=True)
    loss = tf.reduce_mean(real_loss) + tf.reduce_mean(generated_loss)
    return loss


def generator_loss(discr_generated,generated,target):
    L1_loss_lambda=100
    gan_loss=keras.losses.binary_crossentropy(
            tf.ones_like(discr_generated),discr_generated,from_logits=True)
    gan_loss=tf.reduce_mean(gan_loss)
    l1_loss=tf.reduce_mean(tf.abs(target-generated))
    return gan_loss + L1_loss_lambda * l1_loss


def get_facade_dataset():
    path_to_facadestargz = keras.utils.get_file(
            'facades.tar.gz',
            cache_subdir=os.path.expanduser("."),
            origin='https://people.eecs.berkeley.edu/~tinghuiz/projects/pix2pix/datasets/facades.tar.gz',
            extract=True)
    print('Downloading to {}'.format(os.path.dirname(path_to_facadestargz)))
    path_to_train_set = os.path.join(os.path.dirname(path_to_facadestargz), 'facades/train/*.jpg')
    train_ds = tf.data.Dataset.list_files(path_to_train_set)
    train_ds = train_ds.map(lambda filename: load_image(filename, True)).shuffle(400).batch(1)
    path_to_val_set = os.path.join(os.path.dirname(path_to_facadestargz), 'facades/test/*.jpg')
    val_ds = tf.data.Dataset.list_files(path_to_val_set)
    val_ds = val_ds.map(lambda filename: load_image(filename, True)).shuffle(400).batch(1)
    return train_ds, val_ds


def load_image(image_filename, is_train):
    """
    load and preprocess images
    """
    images=tf.io.read_file(image_filename)
    images=tf.image.decode_jpeg(images)
    images=tf.cast(images,tf.float32)
    actual_width=images.shape[1]//2
    drawn=images[:, actual_width:, :] # right half
    real=images[:, :actual_width, :] # left half
    if is_train:
        drawn,real= random_jitter_image_pair(drawn, real)
    else:
        drawn=tf.image.resize(drawn, size=[facades_img_height,facades_img_width])
        real=tf.image.resize(real, size=[facades_img_height,facades_img_width])
    drawn, real = drawn/127.5- 1, real/127.5 - 1
    return tf.concat([drawn,real],axis=2)


def random_jitter_image_pair(drawn,real):
    drawn = tf.image.resize(drawn, [facades_img_height+30, facades_img_width+30])
    real = tf.image.resize(real, [facades_img_height+30, facades_img_width+30])
    stacked = tf.stack([drawn, real], axis=0)
    cropped = tf.image.random_crop(stacked, size=[2, facades_img_height, facades_img_width, 3])
    drawn, real = cropped[0], cropped[1]
    if np.random.random() > 0.5:
        drawn = tf.image.flip_left_right(drawn)
        real = tf.image.flip_left_right(real)
    return real,drawn


class Downsample(keras.Sequential):

    def __init__(self, filters, kernel_size, apply_batchnorm=True):
        init= tf.random_normal_initializer(0., 0.02)
        layers=[
                keras.layers.Conv2D(filters,
                                    kernel_size,
                                    strides=2,
                                    padding='same',
                                    kernel_initializer=init,
                                    use_bias=False)]
        if(apply_batchnorm): layers.append(keras.layers.BatchNormalization())
        layers.append(keras.layers.LeakyReLU())
        super(Downsample, self).__init__(layers)

    def call(self, inputs, training=None, mask=None):
        super(Downsample, self).call(inputs,training,mask)

    def build(self, input_shape=None):
            super(Downsample, self).build(input_shape=input_shape)


class Upsample(keras.Sequential):

    def __init__(self, filters, kernel_size, apply_dropout=False, dropout_rate=0.5):
        super(Upsample, self).__init__()
        init=tf.random_normal_initializer(0., 0.02)
        self.add(keras.layers.Conv2DTranspose(filters,
                                              kernel_size,
                                              strides=2,
                                              padding='same',
                                              kernel_initializer=init,
                                              use_bias=False,
                                              name='up_conv'))
        self.add(keras.layers.BatchNormalization())
        if apply_dropout : self.add( keras.layers.Dropout(dropout_rate))
        self.add(keras.layers.ReLU())

    def call(self, x1, x2, training=None):
        fromInputs=super(Upsample, self).call(x1, training, mask)
        return tf.concat([fromInputs, x2], axis=-1)


class Generator(keras.Sequential):

    def __init__(self):
        init=tf.random_normal_initializer(0., 0.02)
        layers=[
            Downsample( 64,4,apply_batchnorm=False),
            Downsample(128,4),
            Downsample(256,4),
            Downsample(512,4),
            Downsample(512,4),
            Downsample(512,4),
            Downsample(512,4),
            Downsample(512,4),
            Upsample(512, 4, apply_dropout=True),
            Upsample(512, 4, apply_dropout=True),
            Upsample(512, 4, apply_dropout=True),
            Upsample(512, 4),
            Upsample(256, 4),
            Upsample(128, 4),
            Upsample( 64, 4),
            keras.layers.Conv2DTranspose(3,
                             4,
                             strides=2,
                             padding='same',
                             kernel_initializer=init,
                             activation=keras.activations.tanh)
        ]
        super(Generator, self).__init__(layers)


class DiscDownsample(keras.Sequential):

    def __init__(self, filters, kernel_size, apply_batchnorm=True):
        super(DiscDownsample, self).__init__()
        init=tf.random_normal_initializer(0., 0.02)
        self.add(
            keras.layers.Conv2D(filters,
                                size,
                                strides=2,
                                padding='same',
                                kernel_initializer=init,
                                use_bias=False))
        if apply_batchnorm: self.add(keras.layers.BatchNormalization())
        self.add(keras.layers.LeakyReLU())


class Discriminator(keras.Sequential):

    def __init__(self):
        init= tf.random_normal_initializer(0., 0.25)
        layers=[
            DiscDownsample(64,4,False),
            DiscDownsample(128,4),
            DiscDownsample(256,4),
            # need padding to go from (batch, 32,32,256) to (batch, 31,31,512)
            keras.layers.ZeroPadding2D(),
            keras.layers.Conv2D(512, 4, strides=1, kernel_initializer=init, use_bias=False),
            keras.layers.BatchNormalization(),
            keras.layers.LeakyReLU(),
            keras.layers.ZeroPadding2D(),
            keras.layers.Conv2D(1, 4, strides=1, kernel_initializer=init),
        ]
        super(Discriminator, self).__init__(layers)

    def call(self, inputs, training=None, mask=None):
        input,target=inputs
        x= tf.concat([input,target],axis=-1)# (bs, 256, 256, channels*2)
        super(Discriminator, self).call(x, training=training, mask=mask)


if __name__ == '__main__':
    main()