import tensorflow as tf
from tensorflow import keras
import os
import numpy as np
from PIL.Image import fromarray as toimage

def main():
    tf.random.set_seed(4729)
    np.random.seed(4729)
    os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
    assert int(tf.__version__.split('.')[0]) >= 2
    generator_noise_dims=100
    epochs=20000
    batch_size=128
    learning_rate = 0.0002
    is_training= True
    save_example_output_images_per_line =  10
    save_dir='./images'
    os.makedirs(save_dir,exist_ok=True)

    # training data
    dbiterator, mnist_batch_shape = get_mnist_train_and_val_with_dims(batch_size)

    #
    generator=Generator(input_shape=(batch_size,generator_noise_dims))
    generator.summary()
    discriminator=Discriminator(input_shape=(batch_size,28,28,1))
    discriminator.summary()

    train_with_save_example_generated_imgs(batch_size, dbiterator, discriminator, epochs, generator,
                                           generator_noise_dims, is_training, learning_rate,
                                           mnist_batch_shape, save_dir,
                                           save_example_output_images_per_line)




class Generator(keras.Model):
    """
    Generator outputs images given input noise
    """
    def __init__(self, input_shape):
        super(Generator, self).__init__()

        self.n_filters_layer1 = 512
        self.kernel_size_layers2_3 = 4

        # input z vector is [None, 100]
        self.dense1 = keras.layers.Dense(3 * 3 * self.n_filters_layer1)
        self.conv2 = keras.layers.Conv2DTranspose(self.n_filters_layer1 // 2, 3, 2, 'valid')
        self.bn2 = keras.layers.BatchNormalization()
        self.conv3 = keras.layers.Conv2DTranspose(self.n_filters_layer1 // 4, self.kernel_size_layers2_3, 2, 'same')
        self.bn3 = keras.layers.BatchNormalization()
        self.conv4 = keras.layers.Conv2DTranspose(1, self.kernel_size_layers2_3, 2, 'same')
        self.build(input_shape)
        return

    def call(self, inputs, training=None):
        # reshape [b,100] -> [b,3,3,512]
        x= tf.reshape(self.dense1(inputs), shape=[-1, 3, 3, self.n_filters_layer1])
        x= tf.nn.leaky_relu(x)
        x= tf.nn.leaky_relu(self.bn2(self.conv2(x), training=training))
        x= tf.nn.leaky_relu(self.bn3(self.conv3(x), training=training))
        x= tf.tanh(self.conv4(x))
        return x


class Discriminator(keras.Sequential):
    """
    Discriminator return a single logit, to be interpreted as 1=real, 0=fake
    """
    def __init__(self, input_shape):
        self.min_filters=64
        self.kernel_size=4
        layers=[
            keras.layers.Conv2D(self.min_filters, self.kernel_size,2,'same'),
            keras.layers.LeakyReLU(),
            keras.layers.Conv2D(self.min_filters * 2, self.kernel_size, 2, 'same'),
            keras.layers.BatchNormalization(),
            keras.layers.LeakyReLU(),
            keras.layers.Conv2D(self.min_filters*4, self.kernel_size, 2, 'same'),
            keras.layers.BatchNormalization(),
            keras.layers.LeakyReLU(),
            keras.layers.Flatten(),
            keras.layers.Dense(1)
        ]
        super(Discriminator, self).__init__(layers)
        self.build(input_shape)
        return


def save_result(img_array, val_block_size, image_save_path, color_mode):

    def preprocess(img):
        img= ((img + 1.0) * 127.5).astype(np.uint8)
        return img

    preprocessed=preprocess(img_array)
    final_image=np.array([])
    single_row=np.array([])
    for b in range(img_array.shape[0]):

        if single_row.size==0:
            single_row = preprocessed[b,:,:,:]
        else:
            single_row = np.concatenate((single_row, preprocessed[b,:,:,:]), axis=1)

        if(b+1) % val_block_size == 0:
            if final_image.size==0:
                final_image= single_row
            else:
                final_image= np.concatenate((final_image,single_row), axis=0)
            single_row=np.array([])

    if final_image.shape[2] == 1:
        final_image= np.squeeze(final_image, axis=2)
    toimage(final_image, mode=color_mode).save(image_save_path)
        
    

#shorten the sigmoid cross entropy loss calc
def cross_entropy_ones_smoothed(logits, smooth=0.0):
    return tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(
                logits=logits,
                labels=tf.ones_like(logits)*(1.0-smooth)))

def cross_entropy_zeros(logits):
    return tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(
                logits=logits,
                labels=tf.zeros_like(logits)))

def discriminators_loss(generator, discriminator, input_noise, real_image, is_training):
    fake_image=generator(input_noise,is_training)
    d_real_logits=discriminator(real_image, is_training)
    d_fake_logits=discriminator(fake_image, is_training)
    d_loss_real=cross_entropy_ones_smoothed(d_real_logits,smooth=0.1)
    d_loss_fake=cross_entropy_zeros(d_fake_logits)
    loss= d_loss_fake + d_loss_real
    return loss

def generators_loss(generator, discriminator, input_noise, is_training):
    fake_image= generator(input_noise, is_training)
    d_fake_logits=discriminator(fake_image, is_training)
    loss=cross_entropy_ones_smoothed(d_fake_logits, smooth=0.1)
    return loss

def train_with_save_example_generated_imgs(batch_size, dbiterator, discriminator, epochs, generator,
                                           generator_noise_dims, is_training, learning_rate,
                                           mnist_batch_shape, save_dir,
                                           save_example_output_images_per_line):
    #
    d_optimizer = keras.optimizers.Adam(learning_rate=learning_rate, beta_1=0.5)
    g_optimizer = keras.optimizers.Adam(learning_rate=learning_rate, beta_1=0.5)
    print('Training {} epochs...'.format(epochs))
    for epoch in range(1,epochs+1):
        batch_x = next(dbiterator)
        batch_x = tf.reshape(batch_x, shape=mnist_batch_shape)
        batch_x = batch_x * 2.0 - 1  # (0,1) -> (-1,1)
        batch_noise = tf.random.uniform(shape=[batch_size, generator_noise_dims], minval=-1.,
                                        maxval=1.)

        with tf.GradientTape() as tape:
            d_loss = discriminators_loss(generator, discriminator, batch_noise, batch_x,
                                         is_training)
        d_grads = tape.gradient(d_loss, discriminator.trainable_variables)
        d_optimizer.apply_gradients(zip(d_grads, discriminator.trainable_weights))

        with tf.GradientTape() as tape:
            g_loss = generators_loss(generator, discriminator, batch_noise, is_training)
        g_grads = tape.gradient(g_loss, generator.trainable_weights)
        g_optimizer.apply_gradients(zip(g_grads, generator.trainable_weights))

        if epoch % 2 == 0: print('.', end='')
        if epoch % 100 == 0:
            print()
            print(epoch, 'D loss: {:.3f}'.format(float(d_loss)), 'g loss:{:.3f}'.format(float(g_loss)))
            example_noise = np.random.uniform(-1, 1, size=(
                save_example_output_images_per_line ** 2, generator_noise_dims))
            fake_image = generator(example_noise, training=False)
            image_savepath = os.path.join(save_dir, 'gan-example-{:04d}.png'.format(epoch + 1))
            save_result(fake_image.numpy(), save_example_output_images_per_line, image_savepath,
                        color_mode='L')


def get_mnist_train_and_val_with_dims(batch_size):
    (xt, yt), (xv, yv) = keras.datasets.mnist.load_data()
    xt = xt.astype(np.float32) / 255.
    db = tf.data.Dataset.from_tensor_slices(xt).shuffle(batch_size * 4).batch(batch_size).repeat()
    dbiterator = iter(db)
    mnist_batch_shape = [-1, 28, 28, 1]
    return dbiterator, mnist_batch_shape


if __name__ == '__main__':
    main()
