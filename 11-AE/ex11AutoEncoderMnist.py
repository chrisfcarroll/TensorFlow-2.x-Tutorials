import os
import numpy as np
from numpy import ndarray
import tensorflow as tf
from PIL import Image
from matplotlib import pyplot as plt
from tensorflow import keras, Tensor
from tensorflow.keras.models import Model

tf.random.set_seed(22)
np.random.seed(22)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
assert tf.__version__ > '2'
mnist_image_size = 28 * 28
hidden_representation_dim = 20
num_epochs = 55
batch_size = 100
learning_rate = 1e-3
progress_interval=50

def load_mnist_as_float32uint8(verbose=True) -> ((ndarray, ndarray), (ndarray, ndarray)):
    (xt, yt), (xv, yv) = keras.datasets.mnist.load_data()
    xt, xv = xt.astype(np.float32) / 255., xv.astype(np.float32) / 255.
    if verbose: print('mnist train={} val={}'.format(
            (xt.shape, yt.shape),(xv.shape, yv.shape)))
    return (xt, yt), (xv, yv)


(xt, yt), (xv, yv) = load_mnist_as_float32uint8()
train_ds = tf.data.Dataset.from_tensor_slices(xt).shuffle(batch_size * 5).batch(batch_size)
num_batches = xt.shape[0] // batch_size


def collect_epoch_demo_images(
        model:Model,
        input_batch:Tensor,
        shape:tuple,
        epochN:int,
        save_in_dir:str= 'images',
        display_images=True):
    batch_size= (input_batch.shape[0] // 2) * 2
    display_x= int(np.math.sqrt(batch_size))
    display_y= int(batch_size/display_x)
    width= shape[0] ; height = shape[1]
    assert input_batch.shape[1] == width * height
    half_batch= input_batch[:batch_size//2]
    logits=model(half_batch)
    activations=tf.nn.sigmoid(logits)
    imgs_shape=[-1, width, height]
    output_imgs=tf.reshape(activations, imgs_shape).numpy()
    half_batch_images=tf.reshape(half_batch, imgs_shape ).numpy()
    concat_imgs = (tf.concat([half_batch_images, output_imgs], axis=0).numpy() * 255
                    ).astype(np.uint8)
    i=0
    demo_batch_images = Image.new('L', (width*display_x, height*display_y))
    for x in range(0,width*display_x,width):
        for y in range(0,height*display_y,height):
            img=Image.fromarray(concat_imgs[i], mode='L')
            demo_batch_images.paste(img, (x, y))
            i+=1
    if save_in_dir is not None:
        demo_batch_images.save('{}/demo_images_epoch{}.png'.format(save_in_dir,epochN))
    if display_images is True:
        plt.imshow(np.asarray(demo_batch_images))
        plt.show()

class AE(tf.keras.Model):

    def __init__(self, image_size, hidden_size=20):
        super(AE, self).__init__()
        self.fc1 = keras.layers.Dense(512)
        self.fc2 = keras.layers.Dense(hidden_size)
        self.fc3 = keras.layers.Dense(512)
        self.fc4 = keras.layers.Dense(image_size)

    def encode(self, input: Tensor) -> Tensor:
        a = tf.nn.relu(self.fc1(input))
        a = self.fc2(a)
        return a

    def decode_logits(self, compressed_representation: Tensor) -> Tensor:
        a = tf.nn.relu(self.fc3(compressed_representation))
        a = self.fc4(a)
        return a

    def decode(self, compressed_representation: Tensor) -> Tensor:
        return tf.nn.sigmoid(self.decode_logits(compressed_representation))

    def call(self, input, training=None, mask=None) -> Tensor:
        compressed_representation = self.encode(input)
        reconstructed_representation = self.decode_logits(compressed_representation)
        return reconstructed_representation


model = AE(mnist_image_size, hidden_representation_dim)
model.build(input_shape=(4, mnist_image_size))
model.summary()
optimizer = keras.optimizers.Adam(learning_rate)

for epoch in range(1, num_epochs+1):
    for step, x in enumerate(train_ds):
        x=tf.reshape(x, [-1, mnist_image_size])
        with tf.GradientTape() as tape:
            reconstructed_logits=model(x)
            loss=tf.reduce_sum(
                tf.nn.sigmoid_cross_entropy_with_logits(labels=x,logits=reconstructed_logits)
            ) / batch_size
        gradients= tape.gradient( loss, model.trainable_weights)
        gradients,__=tf.clip_by_global_norm(gradients,15)
        optimizer.apply_gradients(zip(gradients,model.trainable_weights))
        if (step+1) % progress_interval == 0:
            print('Epoch {:2}/{} Step {:4}/{} : Loss {:4}'
                  ''.format(epoch,num_epochs,step+1,num_batches,loss))

    collect_epoch_demo_images(model,x,(28,28),epoch)
