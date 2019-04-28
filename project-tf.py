# -*- coding: utf-8 -*-

from __future__ import absolute_import, division, print_function

import tensorflow as tf
import numpy as np
import time
from scipy import misc

tf.logging.set_verbosity(tf.logging.INFO)


def _parse_function(proto):
    # label and image are stored as bytes but could be stored as 
    # int64 or float64 values in a serialized tf.Example protobuf.
    tfrecord_features = tf.parse_single_example(proto,
                        features={
                            'label': tf.FixedLenFeature([], tf.int64),
                            'shape': tf.FixedLenFeature([], tf.string),
                            'image': tf.FixedLenFeature([], tf.string),
                        }, name='features')
    # image was saved as uint8, so we have to decode as uint8.
    image = tf.decode_raw(tfrecord_features['image'], tf.uint8)
    image = tf.image.convert_image_dtype(image, dtype=tf.float32)
    shape = tf.decode_raw(tfrecord_features['shape'], tf.int32)
    # the image tensor is flattened out, so we have to reconstruct the shape
    image = tf.reshape(image, shape)
    label = tfrecord_features['label']
    label = tf.one_hot(label, 10)
    return image, label


def create_dataset(file_names, batch_size=64, prefetch=2):
    # This works with arrays as well
    dataset = tf.data.TFRecordDataset(file_names)
    
    # Maps the parser on every filepath in the array. You can set the number of parallel loaders here
    dataset = dataset.map(_parse_function, num_parallel_calls=8)
    
    # This dataset will go on forever
    dataset = dataset.repeat()
    
    # Set the number of datapoints you want to load and shuffle 
    #dataset = dataset.shuffle(2000)
    
    # Set the batchsize
    dataset = dataset.batch(batch_size)
    
    dataset = dataset.prefetch(buffer_size=prefetch)
    
    return dataset


def inference_dataset(image_names):
    
    num_images = len(image_names)
    
    images = np.zeros([num_images, 480,640], dtype=np.uint8)
    for i in range(num_images):
        image = misc.imread(num_images)
        image = np.asarray(image, np.uint8)
        images[i,:,:] = image
    
    img_tensor = tf.image.convert_image_dtype(images, dtype=tf.float32)
    dataset = tf.data.Dataset.from_tensor_slices(images))
    
    
    
    # This dataset will go on forever
    dataset = dataset.repeat()
    
    # Set the number of datapoints you want to load and shuffle 
    #dataset = dataset.shuffle(2000)
    
    # Set the batchsize
    dataset = dataset.batch(batch_size)
    
    dataset = dataset.prefetch(buffer_size=prefetch)
    
    return dataset

train_dataset = create_dataset('train.tfrecords')
test_dataset = create_dataset('test.tfrecords')

iterator = tf.data.Iterator.from_structure(train_dataset.output_types,
                                               train_dataset.output_shapes)

batch_x, batch_y  = iterator.get_next()


xav_init = tf.contrib.layers.xavier_initializer()
# make datasets that we can initialize separately, but using the same structure via the common iterator
iterator_training_init_op = iterator.make_initializer(train_dataset)
iterator_test_init_op = iterator.make_initializer(test_dataset)

# Input Layer
inputs = tf.reshape(batch_x, [-1, 480, 640, 3])

# Convolutional Layer #1

conv1 = tf.layers.conv2d(inputs=inputs, filters=32, kernel_size=[5, 5], padding="same", activation=tf.nn.relu, kernel_initializer=xav_init)

# Pooling Layer #1
pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[3, 3], strides=3)

# Convolutional Layer #2 and Pooling Layer #2
conv2 = tf.layers.conv2d(inputs=pool1, filters=64, kernel_size=[5, 5], padding="same", activation=tf.nn.relu, kernel_initializer=xav_init)
pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[3, 3], strides=3)

conv3 = tf.layers.conv2d(inputs=pool2, filters=32, kernel_size=[5, 5], padding="same", activation=tf.nn.relu, kernel_initializer=xav_init)
pool3 = tf.layers.max_pooling2d(inputs=conv3, pool_size=[2, 2], strides=2)

conv4 = tf.layers.conv2d(inputs=pool3, filters=16, kernel_size=[5, 5], padding="same", activation=tf.nn.relu, kernel_initializer=xav_init)
pool4 = tf.layers.max_pooling2d(inputs=conv4, pool_size=[2, 2], strides=2)
print(pool4.shape)
# Dense Layer
pool3_flat = tf.reshape(pool4, [-1, 13*17*16])
dense1 = tf.layers.dense(inputs=pool3_flat, units=512, activation=tf.nn.relu, kernel_initializer=xav_init)
dense2 = tf.layers.dense(inputs=dense1, units=128, activation=tf.nn.relu, kernel_initializer=xav_init)
#dropout = tf.layers.dropout(inputs=dense, rate=0.4, training=True)

    # Logits Layer
logits = tf.layers.dense(inputs=dense2, units=10, kernel_initializer=xav_init)

predictions = tf.nn.softmax(logits)

cost = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=batch_y) )

optimizer = tf.train.AdamOptimizer().minimize(cost)

correct_pred = tf.equal(tf.argmax(predictions, 1), tf.argmax(batch_y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# to see if gpu used
#sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))

sess = tf.Session()
sess.run(tf.global_variables_initializer())
sess.run(iterator_training_init_op)

n_iterations = 200

start_time = time.time()

for i in range(n_iterations):

    _, train_loss, train_accuracy = sess.run([optimizer, cost, accuracy])
    print("Iteration:" + str(i) + "\t| Loss =" + str(train_loss) + "\t| Accuracy =" + str(train_accuracy))


train_time = time.time() - start_time

print("training time: " + str(train_time))

# switch iterator to test set
sess.run(iterator_test_init_op)

start_time = time.time()
test_accuracy = 0
for i in range(52):
    test_accuracy += sess.run(accuracy)
    
test_time = start_time - time.time()
print("\nAccuracy on test set:", test_accuracy/52)
test_time = time.time() - start_time

print("test time: " + str(test_time))
