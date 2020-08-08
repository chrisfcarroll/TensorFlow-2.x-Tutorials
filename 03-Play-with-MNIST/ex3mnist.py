import numpy as np
from numpy import ndarray
import tensorflow as tf
from tensorflow import keras, data
from tensorflow.keras import datasets, layers, models, optimizers, metrics, Sequential
from tensorflow.data import Dataset

xs:ndarray
ys:ndarray
(xs,ys),__= datasets.mnist.load_data() #throws away _=(xtrain,ytrain)
print('dataset:', xs.shape, ys.shape, xs.min(), xs.max())

xs= xs.astype(np.float32) /255
db=tf.data.Dataset.from_tensor_slices((xs,ys)).batch(32).repeat(10)

net=keras.Sequential(
    [layers.Dense(256,activation='relu'),
    layers.Dense(256,activation='relu'),
    layers.Dense(256,activation='relu'),
    layers.Dense(10,activation='relu')] )

net.build(input_shape=(None,28*28))
net.summary()
optimizer=optimizers.SGD(lr=0.01)
accuracy_meter=metrics.Accuracy()

for step, (x,y) in enumerate(db):
    with tf.GradientTape() as tape:
        x=tf.reshape(x, (-1,28*28))
        out=net(x)
        y_onehot=tf.one_hot(y, depth=10)
        loss=tf.square( out - y_onehot)
        loss= tf.reduce_sum(loss)/32
    accuracy_meter.update_state(tf.argmax(out,axis=1), y)
    grads=tape.gradient(loss,net.trainable_variables)
    if step % 1000==0:
        print(step, 'loss:', float(loss), 'accuracy:', accuracy_meter.result().numpy()*100,'%')




