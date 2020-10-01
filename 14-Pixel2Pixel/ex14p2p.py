from typing import Tuple
import numpy as np
from numpy import ndarray, math
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, optimizers, metrics
from tensorflow.keras.models import Model, Sequential
from tensorflow.python.data import Dataset
from matplotlib import pyplot as plt
import os
import time
from gds import *

assert tf.__version__ >='2'; os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.random.set_seed(3426) ; np.random.seed(3426)
P2P_Adam_Learning_Rate=2e-4
P2P_Adam_Beta_1=0.5
P2P_Batch_Size=1
Facades_Img_Width=256
Facades_Img_Height=256


def main():
    generator : keras.Model = Generator()
    generator.build(
            input_shape=(P2P_Batch_Size, Facades_Img_Width, Facades_Img_Height, 3))
    generator.optimizer=keras.optimizers.Adam(
            learning_rate=P2P_Adam_Learning_Rate,
            beta_1=P2P_Adam_Beta_1)
    generator.summary()

    discriminator: keras.Model = Discriminator()
    discriminator.build(
            input_shape=[
                (P2P_Batch_Size, Facades_Img_Width, Facades_Img_Height, 3),
                (P2P_Batch_Size, Facades_Img_Width, Facades_Img_Height, 3)])
    discriminator.optimizer=keras.optimizers.Adam(
            learning_rate=P2P_Adam_Learning_Rate,
            beta_1=P2P_Adam_Beta_1)
    discriminator.summary()

    train_ds, val_ds = get_facade_dataset()

    train(generator, discriminator, train_ds, val_ds)


def train(
        generator:Generator,
        discriminator:Discriminator,
        train_ds:Dataset,
        val_ds:Dataset,
        epochs=100):
    print('Training for {} epochs ...'.format(epochs))
    for epoch in range(1,epochs+1):
        start=time.time()
        for step,inputs in enumerate(train_ds):
            input,target=tf.split(inputs,num_or_size_splits=[3,3],axis=3)
            #
            # imshow_n_images( input[0]*0.5 + 0.5,target[0]*0.5 + 0.5 )
            print('input.shape={}, target.shape={}'.format(input.shape, target.shape))
            plt.savefig('images/examplepair{:04d}.png'.format(epoch))
            #
            with tf.GradientTape() as gen_tape, tf.GradientTape() as discr_tape:
                gen_output=generator(input,training=True)
                discr_real=discriminator([input,target], training=True)
                discr_fake=discriminator([input,gen_output], training=True)
                gen_loss= generator.generator_loss(discr_fake,gen_output,target)
                discr_loss=discriminator.discriminator_loss(discr_real,discr_fake)
            gen_gradients= gen_tape.gradient(gen_loss,generator.trainable_weights)
            generator.optimizer.apply_gradients(
                    zip(gen_gradients,generator.trainable_weights))
            discr_gradients=discr_tape.gradient(discr_loss,discriminator.trainable_weights)
            discriminator.optimizer.apply_gradients(
                    zip(discr_gradients,discriminator.trainable_weights))
            if epoch%100==0:
                print('epoch {}, batch {}, gen/discr lossess {}'.\
                            format(epoch, step, (gen_loss,discr_loss)))
            if epoch in [1,2,4,10,30,50,80]:
                for inputs in val_ds:
                    input,target=tf.split(inputs,num_or_size_splits=[3,3], axis=3)
                    generate_and_show_images(generator,input,target,epoch)
        print('epoch {} took {}'.format(epoch, time.time()-start))

    for inputs in val_ds:
        input,target=tf.split(inputs,num_or_size_splits=[3,3], axis=3)
        generate_and_show_images(generator,input,target,epochs+1)


def imshow_n_images(*images, rows=1):
    n= len(images)
    cols = math.ceil( n // rows)
    plt.figure(figsize=(rows, cols))
    for i,img in enumerate(images,start=1):
        plt.subplot(rows,cols,i)
        plt.imshow(img)
    plt.show()


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


def get_facade_dataset():
    path_to_facadestargz = keras.utils.get_file(
            'facades.tar.gz',
            origin=
            'https://people.eecs.berkeley.edu/~tinghuiz/projects/pix2pix/datasets/facades.tar.gz',
            extract=True)
    print('Downloading to {}'.format(os.path.dirname(path_to_facadestargz)))
    path_to_train_set = os.path.join(os.path.dirname(path_to_facadestargz), 'facades/train/*.jpg')
    path_to_val_set = os.path.join(os.path.dirname(path_to_facadestargz), 'facades/test/*.jpg')
    train_ds1 = tf.data.Dataset.list_files(path_to_train_set)
    train_ds2= train_ds1.map(tf.io.read_file)
    train_ds3= train_ds2.map(tf.image.decode_jpeg) # jpegs are 512*256 * 3 channel, int8
    train_ds4= train_ds3.map(split_image_in_two) #split into a pair of 256*256
    train_ds5= train_ds4.map(lambda d,r: # resizing also converts int to float
                             (tf.image.resize(d, [Facades_Img_Height + 30, Facades_Img_Width + 30]),
                              tf.image.resize(r, [Facades_Img_Height + 30, Facades_Img_Width + 30])))
    train_ds6 = train_ds5.map(cropped_image_pair)
    train_ds7 = train_ds6.map(
            lambda d,r: (d,r) \
                if np.random.random()>0.5 \
                else (tf.image.flip_left_right(d),tf.image.flip_left_right(r)))
    train_ds8 = train_ds7.map(lambda d,r: (d/127.5 -1, r/127.5 -1) ) #map [0,255]-> [-1,1]
    train_ds = train_ds8.map(lambda d,r: tf.concat((d,r),axis=2))
    train_ds = train_ds.shuffle(400).batch(1)

    val_ds = tf.data.Dataset.list_files(path_to_val_set)
    val_ds= val_ds.map(tf.io.read_file)
    val_ds= val_ds.map(tf.image.decode_jpeg) # jpegs are 512*256 * 3 channel, int8
    val_ds = val_ds.map(split_image_in_two) # split into a pair of 256*256
    val_ds= val_ds.map(lambda d,r:  # resizing also converts int to float
                             (tf.image.resize(d, [Facades_Img_Height, Facades_Img_Width]),
                              tf.image.resize(r, [Facades_Img_Height, Facades_Img_Width])))
    val_ds = val_ds.map(lambda d,r: (d/127.5 -1, r/127.5 -1) ) #map [0,255]-> [-1,1]
    val_ds = val_ds.map(lambda d,r: tf.concat((d,r),axis=2))
    val_ds = val_ds.shuffle(400).batch(1)
    return train_ds, val_ds

@tf.function
def random_jitter_image_pair(drawn:tf.Tensor,real:tf.Tensor):
    if drawn.shape.rank>0:
        drawn = tf.image.resize(drawn, [Facades_Img_Height + 30, Facades_Img_Width + 30])
        real = tf.image.resize(real, [Facades_Img_Height + 30, Facades_Img_Width + 30])
        drawn, real = cropped_image_pair(drawn, real)
        if np.random.random() > 0.5:
            drawn = tf.image.flip_left_right(drawn)
            real = tf.image.flip_left_right(real)
    return drawn,real

@tf.function
def cropped_image_pair(drawn, real):
    stacked = tf.stack([drawn, real], axis=0)
    cropped = tf.image.random_crop(stacked, size=[2, Facades_Img_Height, Facades_Img_Width, 3])
    drawn, real = cropped[0], cropped[1]
    return drawn, real


@tf.function
def split_image_in_two(image_pair, halfwidth=Facades_Img_Width):
    drawn = image_pair[:, halfwidth:, :]  # right half
    real = image_pair[:, :halfwidth, :]  # left half
    return drawn, real


if __name__ == '__main__':
    main()