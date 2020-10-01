import  tensorflow as tf
import  numpy as np
from    tensorflow import keras
import os, warnings


class Generator(keras.Sequential):

    L1_Loss_Lambda=100

    @staticmethod
    def generator_loss(discr_generated, generated, target):
        gan_loss=keras.losses.binary_crossentropy(
                tf.ones_like(discr_generated),discr_generated,from_logits=True)
        gan_loss=tf.reduce_mean(gan_loss)
        l1_loss=tf.reduce_mean(tf.abs(target-generated))
        return gan_loss + Generator.L1_Loss_Lambda * l1_loss

    def __init__(self):
        super(Generator, self).__init__()
        initializer = tf.random_normal_initializer(0., 0.02)
        self.add( Downsample(64, 4, apply_batchnorm=False) ),
        self.add( Downsample(128, 4) ),
        self.add( Downsample(256, 4) ),
        self.add( Downsample(512, 4) ),
        self.add( Downsample(512, 4) ),
        self.add( Downsample(512, 4) ),
        self.add( Downsample(512, 4) ),
        self.add( Downsample(512, 4) ),
        self.add( Upsample(512, 4, apply_dropout=True) ),
        self.add( Upsample(512, 4, apply_dropout=True) ),
        self.add( Upsample(512, 4, apply_dropout=True) ),
        self.add( Upsample(512, 4) ),
        self.add( Upsample(256, 4) ),
        self.add( Upsample(128, 4) ),
        self.add( Upsample(64, 4) ),
        self.add( keras.layers.Conv2DTranspose(3, (4, 4),
                                                    strides=2,
                                                    padding='same',
                                                    kernel_initializer=initializer))
        self.add(keras.layers.Activation(tf.nn.tanh))

class Discriminator(keras.Sequential):

    @staticmethod
    def discriminator_loss(discr_real,discr_generated):
        # [1,30,30,1] with [1,30,30,1]
        real_loss=keras.losses.binary_crossentropy(
                tf.ones_like(discr_real), discr_real, from_logits=True)
        generated_loss=keras.losses.binary_crossentropy(
                tf.zeros_like(discr_generated), discr_generated, from_logits=True)
        loss = tf.reduce_mean(real_loss) + tf.reduce_mean(generated_loss)
        return loss

    def __init__(self):

        initializer = tf.random_normal_initializer(0., 0.02)
        layers= [
            Downsample(64, 4, False),
            Downsample(128, 4),
            Downsample(256, 4),
            # we are zero padding here with 1 because we need our shape to
            # go from (batch_size, 32, 32, 256) to (batch_size, 31, 31, 512)
            keras.layers.ZeroPadding2D(),
            keras.layers.Conv2D(512, (4, 4),
                              strides=1,
                              kernel_initializer=initializer,
                              use_bias=False),
            keras.layers.BatchNormalization(),
            keras.layers.LeakyReLU(),
            # shape change from (batch_size, 31, 31, 512) to (batch_size, 30, 30, 1)
            keras.layers.ZeroPadding2D(),
            keras.layers.Conv2D(1, (4, 4),
                 strides=1,
                 kernel_initializer=initializer)
            # don't add a sigmoid activation here since
            # the loss function expects raw logits.
        ]
        super(Discriminator, self).__init__(layers)
        # self.model=keras.models.Sequential(layers)

    def call(self, inputs, training=None):
        inp, target = inputs
        # concatenating the input and the target)
        x = tf.concat([inp, target], axis=-1)  # (bs, 256, 256, channels*2)
        x = super(Discriminator, self).call(x)
        return x

    def build(self, input_shape):
        if tf.__version__ <= "2.3":
            warnings.warn(UserWarning('This call relies on TensorFlow 2.3 but you have {}.'
                                      'You can work round this by using call( <tensor> ) '
                                      'instead of build( <tensorshape> )'\
                          .format(tf.__version__)))
        inp,targets=input_shape
        concated_shape=[inp[0], inp[1], inp[2], inp[3]+targets[3] ]
        # self.model.build(concated_shape)
        super(Discriminator, self).build(concated_shape)



class Downsample(keras.Sequential):

    def __init__(self, filters, size, apply_batchnorm=True):
        initializer = tf.random_normal_initializer(0., 0.02)
        layers=[
             keras.layers.Conv2D(filters,
                                (size, size),
                                strides=2,
                                padding='same',
                                kernel_initializer=initializer,
                                use_bias=False)]
        if apply_batchnorm: layers.append(keras.layers.BatchNormalization())
        layers.append(keras.layers.LeakyReLU())
        super(Downsample, self).__init__(layers)


class Upsample(keras.Sequential):

    def __init__(self, filters, size, apply_dropout=False):
        super(Upsample, self).__init__()
        initializer = tf.random_normal_initializer(0., 0.02)
        self.add(keras.layers.Conv2DTranspose(filters,
                           (size, size),
                           strides=2,
                           padding='same',
                           kernel_initializer=initializer,
                           use_bias=False))
        self.add(keras.layers.BatchNormalization())
        if apply_dropout: self.add(keras.layers.Dropout(0.5))

    def call(self, inputs, x2=None, **kwargs):
        x= super(Upsample, self).call(inputs, **kwargs)
        if x2==None:
            return x
        else:
            x = tf.concat([x, x2], axis=-1)
            return x
