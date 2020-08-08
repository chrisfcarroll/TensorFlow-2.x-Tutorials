import numpy as np
import tensorflow as tf
from tensorflow import compat

tf.compat.v1.disable_eager_execution()
example_X=np.reshape(np.random.randn(784, ), (-1, 784))
example_Y=np.reshape(np.random.randint(1, size=(10)), (-1, 10))
Relu_Layer= compat.v1.keras.layers.Dense(100, input_shape=(784,), activation=tf.nn.relu)
Logit_Layer=compat.v1.keras.layers.Dense(10, input_shape=(100,))
V1SGDTrainer= compat.v1.train.GradientDescentOptimizer(1e-2)

v1inputs= compat.v1.placeholder(tf.float32, shape=[None, 784])
v1labels= compat.v1.placeholder(tf.int16, shape=[None,10])
v1hidden= Relu_Layer(v1inputs)
v1logits=Logit_Layer(v1hidden)
crossentropy= compat.v1.nn.softmax_cross_entropy_with_logits(logits=v1logits, labels=v1labels)
loss=tf.reduce_mean(crossentropy)
train_step=V1SGDTrainer.minimize(loss, var_list=Relu_Layer.weights+Logit_Layer.weights)
session= compat.v1.InteractiveSession()
session.run(compat.v1.global_variables_initializer())
for step in range(10):
    session.run(train_step, feed_dict={v1inputs:example_X, v1labels:example_Y})

