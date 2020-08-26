import sys

import numpy as np
from numpy import ndarray
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, optimizers, metrics, datasets
from tensorflow.data import Dataset
import os
import argparse
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

argparser= argparse.ArgumentParser(prog="ex6TrainCifarVGG16")
argparser.add_argument('ignoredfilename',type=str,default='')
argparser.add_argument('--train-dir', type=str, default='./cifar10_train',
                       help='Directory for event logs and checkpoints')
argparser.add_argument('--max-steps', type=int, default=999999,
                       help="Number of batches to run.")
argparser.add_argument('--log-device-placement', action='store_true',
                       help="If true, log the device placement")
argparser.add_argument('--progress-frequency',type=int, default=20,help='show progress every _ steps')
opts=argparser.parse_args(sys.argv)


def normalise_to_meanzero_stdevone(x_train:tf.Tensor, x_val:tf.Tensor):
        x_train = x_train/255.
        x_val = x_val/255.
        mean=np.mean(x_train,axis=(0,1,2,3))
        std=np.std(x_train,axis=(0,1,2,3))
        print(f'test data mean={mean}, σ={std}')
        x_train= (x_train - mean)/ (std + 1e-7)
        x_val= (x_val - mean)/ (std + 1e-7)
        return x_train,x_val

def cast_to_float32int32(x:tf.Tensor,y:tf.Tensor):
    return tf.cast(x,tf.float32), tf.cast(y,tf.int32)

def compute_loss(logits,labels):
    return tf.reduce_mean(
        tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits,labels=labels)
    )


tf.random.set_seed(22)
print('loading data ...')
x:tf.Tensor;y:tf.Tensor;x_val:tf.Tensor;y_val:tf.Tensor
(x,y), (x_val, y_val)= datasets.cifar10.load_data()
print(f'(x,y).shape={(x.shape,y.shape)}. (x_val,y_val).shape={(x_val.shape,y_val.shape)}')

x, x_val= normalise_to_meanzero_stdevone(x, x_val)
train_ds=tf.data.Dataset.from_tensor_slices((x,y)).map(cast_to_float32int32).shuffle(x.shape[0]).batch(256)
val_ds=tf.data.Dataset.from_tensor_slices((x_val,y_val)).map(cast_to_float32int32).shuffle(x_val.shape[0]).batch(256)
print('. . . loaded.')


model= keras.models.Sequential([
    keras.layers.Reshape(target_shape=(32*32*3,), input_shape=(32,32,3)),
    keras.layers.Dense(100),
    keras.layers.Dense(10)
    ])
criterion=keras.losses.CategoricalCrossentropy(from_logits=True)
accuracy=keras.metrics.CategoricalAccuracy()
optimizer=optimizers.Adam(learning_rate=0.0001)

print('starting...')
for epoch in range(10):
    for step,(x,y) in enumerate(train_ds):
        y= tf.squeeze(y, axis=1)
        y= tf.one_hot(y, depth=10)
        with tf.GradientTape() as tape:
            logits=model(x)
            loss=criterion(y, logits)
            accuracy.update_state(y, logits)
        grads= tape.gradient(loss, model.trainable_weights)
        grads= [ tf.clip_by_norm(g,15) for g in grads]
        optimizer.apply_gradients(zip(grads,model.trainable_weights))

        if step<2 or step % opts.progress_frequency==0:
            print(f'epoch {epoch:3} step {step:6} | loss={loss:4.2f} | accuracy={int(accuracy.result()*100):2}%')
            accuracy.reset_states()

    if epoch % 1 == 0:
        validation_accuracy=keras.metrics.CategoricalAccuracy()
        for x,y in val_ds:
            y=tf.squeeze(y,axis=1)
            y=tf.one_hot(y,depth=10)
            logits=model.predict(x)
            validation_accuracy.update_state(y,logits)
        print(f'                                 | accuracy={int(validation_accuracy.result()*100):2}%')
        validation_accuracy.reset_states()


keras.models.save_model(model, 'Cifar10VGG16')
