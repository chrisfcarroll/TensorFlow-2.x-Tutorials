import numpy as np
from numpy import ndarray
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, optimizers, metrics
from tensorflow.python.data import Dataset
from matplotlib import pyplot as plt
import os
import time
from gds import *

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
            origin='https://people.eecs.berkeley.edu/~tinghuiz/projects/pix2pix/datasets/facades.tar.gz',
            extract=True)
    print('Downloading to {}'.format(os.path.dirname(path_to_facadestargz)))
    path_to_train_set = os.path.join(os.path.dirname(path_to_facadestargz), 'facades/train/*.jpg')
    path_to_val_set = os.path.join(os.path.dirname(path_to_facadestargz), 'facades/test/*.jpg')
    ph_str=tf.keras.backend.placeholder(shape=[],dtype=tf.string)
    train_ds = tf.data.Dataset.list_files(path_to_train_set)
    train_ds = train_ds.map(lambda x: tf.py_function(func=load_train_image, inp=[ph_str], Tout=tf.float32)).shuffle(400).batch(1)
    val_ds = tf.data.Dataset.list_files(path_to_val_set)
    val_ds = val_ds.map(lambda x: tf.py_function(func=load_val_image, inp=[ph_str], Tout=tf.float32)).shuffle(400).batch(1)
    return train_ds, val_ds


def load_train_image(image_filename):
    """
    load and preprocess images
    """
    drawn, real = half_images_from_filename(image_filename)
    drawn, real= random_jitter_image_pair(drawn, real)
    drawn, real = drawn/127.5- 1, real/127.5 - 1
    return tf.concat([drawn,real],axis=2)


def load_val_image(image_filename):
    """
    load and preprocess images
    """
    drawn, real = half_images_from_filename(image_filename)
    drawn=tf.image.resize(drawn, size=[facades_img_height,facades_img_width])
    real=tf.image.resize(real, size=[facades_img_height,facades_img_width])
    drawn, real = drawn/127.5- 1, real/127.5 - 1
    return tf.concat([drawn,real],axis=2)


def half_images_from_filename(image_filename):
    images = tf.io.read_file(image_filename[0])
    images = tf.image.decode_jpeg(images)
    images = tf.cast(images, tf.float32)
    print('filename {} shape {}'.format(image_filename[0], images.shape))
    actual_width = images.shape[1] // 2
    drawn = images[:, actual_width:, :]  # right half
    real = images[:, :actual_width, :]  # left half
    return drawn, real


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


if __name__ == '__main__':
    main()