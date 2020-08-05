import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, optimizers, datasets
from PIL import Image
import numpy as np
import asciiartnumpy

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # 0,1,2

def mnist_ds(take=20000,batch=100):
    (x,y), _ = datasets.mnist.load_data()
    ds= tf.data.Dataset.from_tensor_slices( (x,y))
    ds= ds.map(as_float32_and_int64)
    ds=ds.take(take).shuffle(take).batch(batch)
    return ds

def as_float32_and_int64(x,y):
    x= tf.cast(x, tf.float32) / 255.0
    y= tf.cast(y, tf.int64)
    return x,y


model= keras.Sequential([
    layers.Reshape(target_shape=(28*28,), input_shape=(28,28) ),
    layers.Dense(100, activation='relu'),
    layers.Dense(100, activation='relu'),
    layers.Dense(10)
    ])

optimizer= optimizers.Adam()

@tf.function
def compute_loss(logits, labels):
    return tf.reduce_mean(
        tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=logits, labels=labels ))

@tf.function
def compute_accuracy(logits,labels):
    predictions=tf.argmax(logits,axis=1)
    return tf.reduce_mean(
        tf.cast(tf.equal(predictions,labels), tf.float32))

@tf.function
def train_one_step(model : keras.Model, optimizer: optimizers.Optimizer, x, y):

    with tf.GradientTape() as tape:
        logits=model(x)
        loss = compute_loss(logits,y)

    grads= tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads,model.trainable_variables))

    accuracy = compute_accuracy(logits,y)

    return loss,accuracy

def train(epoch, model, optimizer, dataset, showProgressEvery=500):
    loss=0
    accuracy=0
    for step, (x,y) in enumerate(dataset):
        loss, accuracy= train_one_step(model, optimizer, x, y)
        if step % showProgressEvery ==0:
            print('epoch', epoch, ': loss=', loss.numpy(), '- accuracy=', accuracy.numpy())
    return loss,accuracy

def show_example_image_and_prediction(
        model:keras.Model, dataset:tf.data.Dataset, batch=0, example=0
        ) -> (Image,int):
    batch= next(iter(mnist_train_ds.skip(batch)), (None,0))
    (images, labels)=batch
    img_as_image= Image.fromarray((images[example].numpy() * 255).astype(np.uint8), 'L')
    img_shaped_for_input=np.reshape( images[example], (1,28,28))
    yhat= model(img_shaped_for_input)
    prediction= tf.argmax(yhat, axis=1).numpy()[0]
    return img_as_image,prediction

def show_example_asciiart_and_prediction(
        model:keras.Model, dataset:tf.data.Dataset, batch=0, example=0,
        trim_top_and_bottom=False
        ) -> (str,int):
    batch= next(iter(mnist_train_ds.skip(batch)), (None,0))
    (images, labels)=batch
    img_as_ascii_lines= asciiartnumpy.array_to_asciiart_lines(images[example].numpy(), trim_top_and_bottom)
    img_shaped_for_input=np.reshape( images[example], (1,28,28))
    yhat= model(img_shaped_for_input)
    prediction= tf.argmax(yhat, axis=1).numpy()[0]
    return img_as_ascii_lines,prediction

mnist_train_ds= mnist_ds()

for epoch in range(10):
    loss, accuracy= train(epoch, model, optimizer, mnist_train_ds)

print('Last epoch', ': loss=', loss.numpy(), '- accuracy=', accuracy.numpy())

print('example 1st image' )
img,prediction=show_example_asciiart_and_prediction(model,mnist_train_ds)
print('predicted=', prediction, 'actual=','\n','\n'.join(img))

# img,prediction= show_example_image_and_prediction(model,mnist_train_ds)
# img.show()
