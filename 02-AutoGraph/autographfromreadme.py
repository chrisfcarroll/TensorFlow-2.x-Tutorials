import numpy as np
import tensorflow as tf
from tensorflow import keras, compat
from tensorflow.keras import datasets, layers, models, optimizers, metrics
from tensorflow.compat import v1

tf.compat.v1.disable_eager_execution()
X=np.reshape(np.random.randn(784, ), (-1, 784))
y=np.reshape(np.array([0,0,0,0,0,1,0,0,0,0], dtype=np.uint8), (-1, 10))

ReLU_Layer = tf.keras.layers.Dense(100, input_shape=(784,), activation=tf.nn.relu)
Logit_Layer = tf.keras.layers.Dense(10, input_shape=(100,))

SGD_Trainer = v1.train.GradientDescentOptimizer(1e-2)

inputs = v1.placeholder(tf.float32, shape=[None, 784])
labels = v1.placeholder(tf.int16, shape=[None, 10])
hidden = ReLU_Layer(inputs)
logits = Logit_Layer(hidden)
entropy = tf.nn.softmax_cross_entropy_with_logits(
    logits=logits, labels=labels)
loss = tf.reduce_mean(entropy)
train_step = SGD_Trainer.minimize(loss,
    var_list=ReLU_Layer.weights+Logit_Layer.weights)

sess = v1.InteractiveSession()
sess.run(v1.global_variables_initializer())

print('1000 steps in a v1 session')
for step in range(1000):
    sess.run(train_step, feed_dict={inputs:X, labels:y})


SGD_Trainer = tf.optimizers.SGD(1e-2)

@tf.function
def loss_fn(inputs=X, labels=y):
    hidden = ReLU_Layer(inputs)
    logits = Logit_Layer(hidden)
    entropy = tf.nn.softmax_cross_entropy_with_logits(
        logits=logits, labels=labels)
    return tf.reduce_mean(entropy)

print('1000 steps in v2')
for step in range(1000):
    SGD_Trainer.minimize(loss_fn,
        var_list=ReLU_Layer.weights+Logit_Layer.weights)

