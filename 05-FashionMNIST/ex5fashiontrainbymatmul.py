import os
import numpy as np
import tensorflow as tf
import tensorflow.keras
from tensorflow import keras
from tensorflow.keras import models, optimizers, layers
from tensorflow.python.data import Dataset
from tensorflow.python.keras.optimizer_v2.adam import Adam
from tensorflow.python.keras.optimizers import Optimizer


def cast_mnist_to_float32_int64(x,y) -> (tf.Tensor,tf.Tensor):
  x = tf.cast(x, tf.float32) / 255.0
  y = tf.cast(y, tf.int64)
  return x, y

def get_fashion_mnist(batchsize=100) \
        -> (tf.data.Dataset,tf.data.Dataset):
    (x,y),(xv,yv)= tf.keras.datasets.fashion_mnist.load_data()
    print(f'Original shape of training set (x,y)={(x.shape,y.shape)}')
    trainds=tf.data.Dataset\
        .from_tensor_slices((x,y))\
        .map(cast_mnist_to_float32_int64)\
        .take(x.shape[0]).shuffle(x.shape[0]).batch(batchsize)
    valds=tf.data.Dataset\
        .from_tensor_slices((xv,yv))\
        .map(cast_mnist_to_float32_int64)\
        .batch(batchsize)
    return (trainds,valds)


def compute_loss(logits,labels)->np.ndarray:
    return tf.reduce_mean(
        tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=logits,labels=labels))


def compute_accuracy(logits,labels)->np.ndarray:
    predictions=tf.argmax(logits,axis=1)
    return tf.reduce_mean(
        tf.cast(tf.equal(predictions,labels),tf.float32))

def train_one_step(
        model:models.Model,
        optimizer:tf.keras.optimizers.Optimizer,
        x:tf.Tensor,y:tf.Tensor):
    with tf.GradientTape() as tape:
        logits=model(x)
        loss=compute_loss(logits,y)
    grads= tape.gradient(loss,model.trainable_weights)
    optimizer.apply_gradients(zip(grads,model.trainable_weights))
    accuracy=compute_accuracy(logits,y)
    return loss,accuracy

def train(epoch, model:models.Model, optimizer:optimizers.Optimizer, train_ds:Dataset):
    optimizer:Optimizer= optimizers.Adam()
    loss=0.0
    accuracy=0.0
    for step,(x,y) in enumerate(train_ds):
        loss,accuracy=train_one_step(model,optimizer,x,y)
        if step%500==0:
            print(f'epoch {epoch:4} | loss {loss:4.2f} | accuracy {accuracy*100:2}%')
    return loss,accuracy


class FeedForwardNet(layers.Layer):

    activation= tf.nn.relu

    def __init__(self,units):
        """
        :param units: [input_dim, h1_dim, ..., hn_dim, output_dim]
        """
        super(FeedForwardNet, self).__init__()
        for i in range(1,len(units)):
            self.add_weight(name='kernel%d'%i,initializer="glorot_normal", shape=[units[i-1],units[i]])
            self.add_weight(name='bias%d'%i, shape=[units[i]])


    def call(self,x):
        """
        :param x: [b, input_dim]
        :return x
        """
        n_trainables=len(self.trainable_weights)
        x= tf.reshape(x,[-1,28*28])
        for i in range(0, n_trainables, 2):
            a=tf.matmul(x, self.trainable_weights[i])+self.trainable_weights[i+1]
            x= tf.nn.relu(a)
        return a


os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

model= FeedForwardNet([28*28,200,200100,10])
optimizer=optimizers.Adam()
train_ds,val_ds= get_fashion_mnist()
_,(xval,yval)=next(enumerate(val_ds))
n_epochs=20

for param in model.trainable_weights:
    print(f'{param.name}:{param.shape}')

for epoch in range(n_epochs):
    loss,accuracy= train(epoch,model,optimizer,train_ds)
    vlogits=model(xval)
    vloss=compute_loss(vlogits,yval)
    vaccuracy=compute_accuracy(vlogits,yval)
    print(f'Validation > loss {vloss:4.2f} | accuracy {vaccuracy*100:2}%')
print(f'Last epoch | loss {loss:4.2f} | accuracy {accuracy*100:2}%')