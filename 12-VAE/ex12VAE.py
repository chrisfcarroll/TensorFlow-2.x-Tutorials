import os
import numpy as np
from numpy import ndarray
import tensorflow as tf
from tensorflow import Tensor
from tensorflow.keras import models, datasets, layers, optimizers
from PIL import Image
from matplotlib import pyplot as plt

assert int(tf.__version__.split('.')[0]) >= 2
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.random.set_seed(22)
np.random.seed(22)
mnist_img_width=28
mnist_img_height=28
save_model_directory= 'model'
save_imgs_directory= 'images'
if not os.path.exists(save_model_directory): os.makedirs(save_model_directory)
if not os.path.exists(save_imgs_directory): os.makedirs(save_imgs_directory)

train_num_epochs=55
train_batch_size=100
train_learning_rate=1e-3

model_hidden_dim=512
model_latent_dim=20

demo_output_width=int( np.sqrt(train_batch_size) * mnist_img_width)
demo_output_height=int( np.sqrt(train_batch_size) * mnist_img_height)


def load_minst_xtrain_and_xval_unlabelled_as_float32(verbose=True)-> (ndarray,ndarray):
    (xt,yt),(xv,yv) = datasets.fashion_mnist.load_data()
    xt:ndarray;xv:ndarray
    if verbose: print(f'xtrain.shape={xt.shape}, xval.shape={xv.shape}')
    return xt.astype(np.float32)/255. , xv.astype(np.float32)/255.


def create_grid_image_from_images(generative_model,
                                  output_width, output_height, sample_width, sample_height,
                                  save_to_filepath=None, show_plot=True):
    demo_image=Image.new('L', (output_width, output_height))
    demo_index=0
    for i in range(0, output_width, sample_width):
        for j in range(0, output_height, sample_height):
            output_sample=Image.fromarray(generative_model[demo_index], mode='L')
            demo_image.paste(output_sample,(i,j))
            demo_index +=1
    if save_to_filepath is not None: demo_image.save(save_to_filepath)
    if show_plot:
        plt.imshow(np.asarray(demo_image))
        plt.show()
    return demo_image


class VAE(models.Model):

    def __init__(self, input_dims: (int,int), *,hidden_layers_dim:int, latent_space_dim: int):
        super(VAE,self).__init__()
        self.input_dims=input_dims
        self.latent_space_dims=latent_space_dim
        self.hidden_layer_dims=hidden_layers_dim
        self.input_flattened_size = input_dims[0] * input_dims[1]
        # input -> h dimensions -> (mean,variance)
        self.fc1=layers.Dense(hidden_layers_dim)
        self.fc2_mean=layers.Dense(latent_space_dim)
        self.fc2_log_variance=layers.Dense(latent_space_dim)

        # and back (mean,variance) -> h -> reconstructed output
        self.fc4=layers.Dense(hidden_layers_dim)
        self.fc5=layers.Dense(self.input_flattened_size)

    def encode(self,input):
        h=tf.nn.relu(self.fc1(input))
        mean,variance= self.fc2_mean(h), self.fc2_log_variance(h)
        return mean,variance

    def  reparameterize(self,mean,log_variance):
        #the reparameterization trick.
        std=tf.exp(log_variance *.5)
        shape= std.shape if std.shape[0] is not None and std.shape[0]>0 else (1,std.shape[-1])
        random_normal_dist=tf.random.normal(shape)
        return mean + random_normal_dist * std

    def decode_logits(self,z):
        h= tf.nn.relu(self.fc4(z))
        return self.fc5(h)

    def decode(self,z):
        return tf.nn.sigmoid(self.decode_logits(z))

    def call(self,inputs:Tensor,training=None, mask=None) ->(Tensor,Tensor,Tensor):
        mean,log_variance=self.encode(inputs)
        z=self.reparameterize(mean,log_variance)
        decoder_logits=self.decode_logits(z)
        return decoder_logits,mean,log_variance

    def generate_sample_2D_images(self, how_many:int) -> ndarray:
        z=tf.random.normal((how_many, self.latent_space_dims))
        out=self.decode(z)
        out=tf.reshape(out, [-1,self.input_dims[0], self.input_dims[1]])
        out= (out.numpy() * 255).astype(np.uint8)
        return out

    def sigmoid_output(self, inputs, reshaped_2D=True) -> Tensor:
        out_logits,__,__= self.call(inputs)
        out_sigmoid=tf.nn.sigmoid(out_logits)
        if reshaped_2D:
            return tf.reshape(out_sigmoid, [-1,self.input_dims[0],self.input_dims[1]])
        else:
            return out_sigmoid


xt,xv= load_minst_xtrain_and_xval_unlabelled_as_float32()
xt:ndarray;xv:ndarray
model=VAE((mnist_img_width,mnist_img_width), hidden_layers_dim=model_hidden_dim, latent_space_dim=model_latent_dim)
mnist_dims = mnist_img_width * mnist_img_height
model.build(input_shape=(None,mnist_dims))
model.summary()
optimizer= optimizers.Adam(train_learning_rate)
num_batches=xt.shape[0]//train_batch_size
trainds=tf.data.Dataset.from_tensor_slices(xt)\
            .shuffle(train_batch_size*500).batch(train_batch_size)
print(f'Starting on {trainds.cardinality()} batches of {train_batch_size} each ...')

for epoch in range(1,1+train_num_epochs):
    for step,x in enumerate(trainds):
        input=tf.reshape(x,[-1,mnist_dims])
        with tf.GradientTape() as tape:
            reconstruction_logits,mean,log_variance=model(input)
            reconstruction_loss=tf.reduce_sum(
                    tf.nn.sigmoid_cross_entropy_with_logits(
                        labels=input,logits=reconstruction_logits)
            ) / train_batch_size
            kldivergence= tf.reduce_mean(
                    -0.5 * tf.reduce_sum(
                    1.
                    + log_variance
                    - tf.square(mean)
                    - tf.exp(log_variance), axis=-1))
            loss=tf.reduce_mean(reconstruction_loss) + kldivergence
        gradients=tape.gradient(loss,model.trainable_weights)
        for g in gradients: tf.clip_by_norm(g,15)
        optimizer.apply_gradients(zip(gradients,model.trainable_weights))
        if step % 50 == 0: print('.',end='')

    print(f'\nEpoch {epoch}/{train_num_epochs} |'
          f' Reconstruction loss {loss:.4f} KL Divergence {kldivergence:.4f}')

    models.save_model(model, save_model_directory)

    sample_generated_images=model.generate_sample_2D_images(train_batch_size)
    create_grid_image_from_images(
            sample_generated_images,
            demo_output_width, demo_output_height, mnist_img_width, mnist_img_height,
            save_to_filepath=f'{save_imgs_directory}/vae_generated_by_model_epoch_{epoch}.png',
            show_plot=True)

    half_batch=input[:train_batch_size // 2]
    reconstructed_samples= model.sigmoid_output(half_batch, reshaped_2D=True)
    half_batch_as_2D=tf.reshape(half_batch,[-1, mnist_img_width, mnist_img_height])
    demo_imgs= (tf.concat([half_batch_as_2D,reconstructed_samples],axis=0).numpy() * 255).astype(np.uint8)

    create_grid_image_from_images(
            demo_imgs,
            demo_output_width,demo_output_height, mnist_img_width,mnist_img_height,
            save_to_filepath=f'{save_imgs_directory}/vae_reconstructed_from_input_epoch_{epoch}.png',
            show_plot=True)