import numpy as np
import tensorflow as tf
from tensorflow import keras, compat
from tensorflow.keras import datasets, layers, models, optimizers, metrics


tf.compat.v1.disable_eager_execution()
X=np.reshape(np.random.default_rng().standard_normal(784, dtype=np.float32 ), (-1, 784))
y=np.reshape(np.array([0, 0, 0, 0, 0, 1, 0, 0, 0, 0], dtype=np.uint8), (-1, 10))

Relu_Layer= tf.keras.layers.Dense(100, input_shape=(784,), activation=tf.nn.relu)
Logit_Layer=tf.keras.layers.Dense(10, input_shape=(100,))

def v1_loss(X:np.ndarray, y:np.ndarray):
    V1SGDTrainer= compat.v1.train.GradientDescentOptimizer(1e-2)
    placeholderInputs= compat.v1.placeholder(tf.float32, shape=[None, 784])
    placeholderLabels= compat.v1.placeholder(tf.int16, shape=[None,10])
    hidden= Relu_Layer(placeholderInputs)
    logits= Logit_Layer(hidden)
    crossentropy= tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=placeholderLabels)
    loss=tf.reduce_mean(crossentropy)
    train_step=V1SGDTrainer.minimize(loss, var_list=Relu_Layer.weights+Logit_Layer.weights)
    session= compat.v1.InteractiveSession()
    session.run(compat.v1.global_variables_initializer())
    for step in range(100):
        session.run(train_step, feed_dict={placeholderInputs:X, placeholderLabels:y})

@tf.function
def v2_precompiled_loss_fn(inputs=X, labels=y):
    hidden=Relu_Layer(inputs)
    logits=Logit_Layer(hidden)
    cross_entropy= tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels)
    return tf.reduce_mean(cross_entropy)

def v2_loss(X:np.ndarray, y:np.ndarray):
    V2SGDTrainer=tf.optimizers.SGD(1e-2)
    for step in range(100):
        V2SGDTrainer.minimize(v2_precompiled_loss_fn, var_list=Relu_Layer.weights + Logit_Layer.weights)


print('100 steps in a v1 session')
v1_loss(X,y)

print('100 steps in v2')
v2_loss(X,y)

