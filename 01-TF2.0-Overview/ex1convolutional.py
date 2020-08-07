import os
import time
import numpy as np
import tensorflow as tf
from tensorflow import keras, data
from tensorflow.keras import datasets, layers, models, optimizers, metrics
from tensorflow.data import Dataset
from asciiartnumpy import *

os.environ['TF_CPP_MIN_LOG_LEVEL'] ='2'

def get_mnist_dataset(batch_size=100, shuffle_train=True):
    x_train : np.ndarray; x_test : np.ndarray; y_train : np.ndarray; y_test : np.ndarray;
    (x_train, y_train), (x_test,y_test)= datasets.mnist.load_data()
    x_train, x_test = x_train / np.float32(255), x_test / np.float32(255)
    # why?? y_train, y_test = y_train.astype(np.int64), y_test.astype(np.int64)
    train_dataset= tf.data.Dataset.from_tensor_slices((x_train,y_train))
    test_dataset = tf.data.Dataset.from_tensor_slices((x_test,y_test))
    if shuffle_train: train_dataset= train_dataset.shuffle(x_train.shape[0])
    return train_dataset.batch(batch_size), \
           test_dataset.batch(batch_size), \
           x_train.shape[1:]

train_ds,test_ds,(input_x,input_y)= get_mnist_dataset()

model= tf.keras.Sequential([
    layers.Reshape(
        target_shape=[input_x,input_y,1],
        input_shape=(input_x,input_y,)),
    layers.Conv2D(2,5,padding='same', activation=tf.nn.relu),
    layers.MaxPooling2D((2,2),(2,2), padding='same'),
    layers.Conv2D(4,5,padding='same', activation=tf.nn.relu),
    layers.MaxPooling2D((2,2),(2,2), padding='same'),
    layers.Flatten(),
    layers.Dense(32,activation=tf.nn.relu),
    layers.Dropout(rate=0.4),
    layers.Dense(10)])

optimizer= optimizers.SGD(learning_rate=0.01, momentum=0.5)

def train_step(model: keras.Model,
               optimizer: optimizers.Optimizer,
               images: np.ndarray,
               labels: np.ndarray,
               loss_function:keras.losses.Loss,
               accuracy_metric:keras.metrics.Metric,
               loss_metric:tf.keras.metrics.Metric):
    with tf.GradientTape() as tape:
        logits=model(images,training=True)
        loss=loss_function(labels, logits)
        accuracy_metric(labels, logits)
        loss_metric(loss)

    grads=tape.gradient(loss,model.trainable_variables)
    optimizer.apply_gradients(zip(grads,model.trainable_variables))
    return loss


def train(model:keras.Model,
          optimizer:optimizers.Optimizer,
          train_dataset: Dataset,
          log_steps=50):
    """
    Trainms model on `dataset` using `optimizer`.
    :param train_dataset: tf.data.Dataset
    :param model: the `keras.Model` being trained
    :param optimizer: the `optimizers.Optimizer` being used
    :return:
    """
    loss_function=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    mean_loss_metric=metrics.Mean('loss', dtype=tf.float32)
    accuracy_metric=tf.keras.metrics.SparseCategoricalAccuracy()

    for images, labels in train_dataset:
        train_step(model,optimizer,images,labels,loss_function,accuracy_metric,mean_loss_metric)

        if optimizer.iterations % log_steps == 0:
            print('batch:', int(optimizer.iterations),
                  'mean loss:', mean_loss_metric.result().numpy(),
                  'accuracy:', accuracy_metric.result().numpy())
            mean_loss_metric.reset_states()
            accuracy_metric.reset_states()

def test(model:keras.Model, test_dataset:Dataset):
    """
    Evaluate `model` on examples from `dataset`
    :param model: the model being tested
    :param test_dataset: the test dataset (images,labels)
    :param step_no:
    :return:
    """
    loss_function=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    mean_loss_metric=metrics.Mean('loss', dtype=tf.float32)
    accuracy_metric=tf.keras.metrics.SparseCategoricalAccuracy()

    for (images,labels) in test_dataset:
        logits=model(images,training=False)
        mean_loss_metric(loss_function(labels,logits))
        accuracy_metric(labels,logits)

    print('Test set loss: {:0.4f} accuracy {:0.2f}%'.format(
        mean_loss_metric.result(), accuracy_metric.result()*100))

def show_example_asciiart_and_prediction(
        model:keras.Model, dataset:tf.data.Dataset, batch=0, example=0,
        trim_top_and_bottom=False
        ) -> (str,int):
    batch= next(iter(mnist_train_ds.skip(batch)), (None,0))
    (images, labels)=batch
    img_as_ascii_lines= asciiartnumpy.array_to_asciiart_lines(images[example].numpy(), trim_top_and_bottom)
    prediction = predict(example, images, model)
    return img_as_ascii_lines,prediction


def predict(model:keras.Model, image:np.ndarray):
    img_shaped_for_input = np.reshape(image, (1, image.shape[0], image.shape[1]))
    logits = model(img_shaped_for_input)
    prediction = tf.argmax(logits, axis=1).numpy()[0]
    return prediction

def delete_folder(folder_path):
    if tf.io.gfile.exists(folder_path):
        tf.io.gfile.rmtree(folder_path)


Model_Saved_State_Folder= './mnistConvModel'
delete_folder(Model_Saved_State_Folder)

checkpoint_folder=os.path.join(Model_Saved_State_Folder, 'checkpoints')
checkpoint=tf.train.Checkpoint(model=model, optimizer=optimizer)
checkpoint.restore(tf.train.latest_checkpoint(checkpoint_folder))
checkpoint_fileprefix=os.path.join(checkpoint_folder, 'ckpt')

Num_Training_Epochs=0
for i in range(Num_Training_Epochs):
    started_at=time.time()
    train(model, optimizer,train_ds,log_steps=200)
    finished_at=time.time()
    print('Epoch #{} ({} steps) took : {}'.format(
        i+1, optimizer.iterations, finished_at - started_at))
    checkpoint.save(checkpoint_fileprefix)

export_path= os.path.join(Model_Saved_State_Folder,'export')
tf.saved_model.save(model,export_path)
