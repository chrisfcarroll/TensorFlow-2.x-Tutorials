import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, optimizers, datasets
from tensorflow.python.data import Dataset
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.optimizers import Optimizer


def cast_mnist_features_and_labels(x, y):
    return tf.cast(x, tf.float32) / 255.0, tf.cast(y, tf.int32)


def get_mnist_dataset() -> (Dataset, Dataset):
    (xtrain, ytrain), (xval, yval) = datasets.fashion_mnist.load_data()
    dstrain = tf.data.Dataset.from_tensor_slices((xtrain, ytrain))
    dstrain = dstrain \
        .map(cast_mnist_features_and_labels) \
        .take(xtrain.shape[0]).shuffle(xtrain.shape[0]) \
        .batch(100)
    dsval = tf.data.Dataset.from_tensor_slices((xval, yval))
    dsval = dsval.map(cast_mnist_features_and_labels)
    return dstrain, dsval


def compute_loss(logits, labels) -> tf.Tensor:
    return tf.reduce_mean(
        tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=logits, labels=labels))


def compute_accuracy(logits, labels)->float:
    predictions = tf.argmax(logits, axis=1, output_type=tf.int32)
    return tf.reduce_mean(
        tf.cast(tf.equal(predictions, labels), tf.float32))


def train_one_step(model: Model, optimizer: optimizers.Optimizer, x, y)->(tf.Tensor,tf.Tensor):
    with tf.GradientTape() as tape:
        logits = model(x)
        loss = compute_loss(logits, y)
        grads = tape.gradient(loss, model.trainable_weights)
        optimizer.apply_gradients(zip(grads, model.trainable_weights))
        accuracy = compute_accuracy(logits, y)
        return loss, accuracy


def train(epoch, model, optimizer)->(tf.Tensor,tf.Tensor):
    train_ds, __ = get_mnist_dataset()
    loss = 0.0
    accuracy = 0.0
    for step, (x, y) in enumerate(train_ds):
        loss, accuracy = train_one_step(model, optimizer,x,y)
        if step % 500 == 0:
            print(f'epoch {epoch:4} | loss {loss.numpy():.3f} | accuracy {accuracy.numpy()*100:.0f}%')
    return loss, accuracy


os.environ['TF_CPP_MIN_LOG_LEVEL']='1'
(train_ds,val_ds)=get_mnist_dataset()

model= keras.models.Sequential([
        layers.Reshape(target_shape=(28*28,), input_shape=(28,28)),
        layers.Dense(200,activation='relu'),
        layers.Dense(200,activation='relu'),
        layers.Dense(10)
    ])

optimizer= optimizers.Adam()

for epoch in range(20):
    loss,accuracy=train(epoch,model,optimizer)

print(f'last epoch | loss {loss.numpy():.3f} | accuracy {accuracy.numpy()*100:.0f}%')

keras.models.save_model(model, '05fashionbytapeandcalcstep')
