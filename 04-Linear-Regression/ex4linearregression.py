import numpy as np
from numpy import ndarray
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, optimizers, metrics
from tensorflow.data import Dataset
import os
xtrain:ndarray ; ytrain:ndarray; xtest:ndarray; ytest:ndarray
tf.random.set_seed(22)
np.random.seed(22)
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
assert tf.__version__.startswith('2.')

class Regressor(keras.layers.Layer):

    def __init__(self):
        super(Regressor,self).__init__()
        self.w=self.add_weight('shape', [13,1])
        self.b=self.add_weight('shape2',[1])
        print(self.w.shape, self.b.shape)
        print(type(self.w), tf.is_tensor(self.w), self.w.name)
        print(type(self.b), tf.is_tensor(self.b), self.b.name)

    def call(self,x):
        return tf.matmul(x, self.w) + self.b


def get_data() -> ((ndarray,ndarray),(ndarray,ndarray)):
    (xtrain,ytrain), (xtest,ytest)=keras.datasets.boston_housing.load_data()
    xtrain=xtrain.astype(np.float32)
    xtest=xtest.astype(np.float32)
    print(xtrain.shape, ytrain.shape, xtest.shape, ytest.shape)
    return (xtrain,ytrain),(xtest,ytest)

(xtrain,ytrain),(xtest,ytest)=get_data()
dbtrain= tf.data.Dataset.from_tensor_slices((xtrain,ytrain)).batch(64)
dbtest= tf.data.Dataset.from_tensor_slices((xtest,ytest)).batch(ytest.shape[0])
assert ytest.shape[0]==102

model=Regressor()
loss_fn=keras.losses.MeanSquaredError()
optimizer=optimizers.Adam()

for epoch in range(200):
    for step,(x,y) in enumerate(dbtrain):
        with tf.GradientTape() as tape:
            logits=model(x)
            logits=tf.squeeze(logits,axis=1)
            loss=loss_fn(y,logits)
        grads=tape.gradient(loss, model.trainable_weights)
        optimizer.apply_gradients(zip(grads,model.trainable_weights));
    print(epoch,'loss:', loss.numpy())
    if epoch%10==0:
        for x,y in dbtest:
            logits=model(x)
            logits=tf.squeeze(logits,axis=1)
            loss=loss_fn(y,logits)
            print(epoch,'test loss:', loss.numpy())
