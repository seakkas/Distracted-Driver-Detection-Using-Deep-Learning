# -*- coding: utf-8 -*-

from __future__ import absolute_import, division, print_function

import tensorflow as tf
import numpy as np
import time
import sys
tf.logging.set_verbosity(tf.logging.INFO)


def _parse_function(proto):
    tfrecord_features = tf.parse_single_example(proto,
                        features={
                            'label': tf.FixedLenFeature([], tf.int64),
                            'shape': tf.FixedLenFeature([], tf.string),
                            'image': tf.FixedLenFeature([], tf.string),
                        }, name='features')
    # images were saved as uint8, so we have to decode as uint8.
    image = tf.decode_raw(tfrecord_features['image'], tf.uint8)
    image = tf.image.convert_image_dtype(image, dtype=tf.float32)
    shape = tf.decode_raw(tfrecord_features['shape'], tf.int32)
    # the image tensor is flattened out, so we have to reconstruct the shape
    image = tf.reshape(image, shape)
    label = tfrecord_features['label']
    label = tf.one_hot(label, 10)
    return {'feature': image}, label


def read_dataset(file_names, mode, batch_size=64, prefetch=4):
    dataset = tf.data.TFRecordDataset(file_names)
    dataset = dataset.map(_parse_function, num_parallel_calls=48)
    if mode == tf.estimator.ModeKeys.TRAIN:
    	dataset = dataset.repeat()
    #	dataset = dataset.shuffle(512)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(buffer_size=prefetch)
    return dataset



def model_fn(features, labels, mode, params):
	training = (mode == tf.estimator.ModeKeys.TRAIN)
	inputs = tf.reshape(features['feature'], [-1, 240, 320, 3])
	xav_init = tf.contrib.layers.xavier_initializer()

	conv1 = tf.layers.conv2d(inputs=inputs, filters=64, kernel_size=[5, 5], padding="same", activation=tf.nn.relu, kernel_initializer=xav_init)
	pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[3, 3], strides=3)

	conv2 = tf.layers.conv2d(inputs=pool1, filters=128, kernel_size=[5, 5], padding="same", activation=tf.nn.relu, kernel_initializer=xav_init)
	pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[3, 3], strides=3)
	
	conv3 = tf.layers.conv2d(inputs=pool2, filters=128, kernel_size=[5, 5], padding="same", activation=tf.nn.relu, kernel_initializer=xav_init)
	pool3 = tf.layers.max_pooling2d(inputs=conv3, pool_size=[2, 2], strides=2)

	conv4 = tf.layers.conv2d(inputs=pool3, filters=64, kernel_size=[5, 5], padding="same", activation=tf.nn.relu, kernel_initializer=xav_init)
	pool4 = tf.layers.max_pooling2d(inputs=conv4, pool_size=[2, 2], strides=2)

	conv5 = tf.layers.conv2d(inputs=pool4, filters=64, kernel_size=[5, 5], padding="same", activation=tf.nn.relu, kernel_initializer=xav_init)
	pool5 = tf.layers.max_pooling2d(inputs=conv5, pool_size=[2, 2], strides=2)

	conv6 = tf.layers.conv2d(inputs=pool5, filters=32, kernel_size=[5, 5], padding="same", activation=tf.nn.relu, kernel_initializer=xav_init)
	pool6 = tf.layers.max_pooling2d(inputs=conv6, pool_size=[2, 2], strides=2)

	flat = tf.layers.flatten(pool6)
	dense1 = tf.layers.dense(inputs=flat, units=1024, activation=tf.nn.relu, kernel_initializer=xav_init)
	dense2 = tf.layers.dense(inputs=dense1, units=256, activation=tf.nn.relu, kernel_initializer=xav_init)
	logits = tf.layers.dense(inputs=dense2, units=10, kernel_initializer=xav_init)

	y_pred = tf.nn.softmax(logits=logits)
	y_pred_cls = tf.argmax(y_pred, axis=1)

	if mode == tf.estimator.ModeKeys.PREDICT:
		spec = tf.estimator.EstimatorSpec(mode=mode, predictions=y_pred_cls)
	else:
		cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(labels=labels, logits=logits)
		loss = tf.reduce_mean(cross_entropy)
		optimizer = tf.train.RMSPropOptimizer(params["learning_rate"])
		train_op = optimizer.minimize(loss=loss, global_step=tf.train.get_global_step())
		accuracy = tf.metrics.accuracy(labels=tf.argmax(labels, axis=1), predictions=y_pred_cls, name='acc_op')
		metrics = {"accuracy": accuracy}
		tf.summary.scalar('accuracy', accuracy[1])
		spec = tf.estimator.EstimatorSpec(mode=mode,loss=loss,train_op=train_op,
                eval_metric_ops=metrics)
	return spec


if len(sys.argv) != 3:
    print('one or more arguments missing! Please check arguments')
    sys.exit()

# experiment name: when different dataset used, to save model to different dir
exp_name = sys.argv[1]

#train.tfrecords and test.tfrecords file directory
tfrecords_dir = sys.argv[2]



train_input_fn = lambda: read_dataset(file_names= tfrecords_dir +  'train.tfrecords',
        mode=tf.estimator.ModeKeys.TRAIN, batch_size=32)
test_input_fn = lambda: read_dataset(file_names= tfrecords_dir + 'test.tfrecords', mode=tf.estimator.ModeKeys.EVAL,
        batch_size=32)

params = {"learning_rate": 0.001}




# to use multiple GPU
distribution = tf.contrib.distribute.MirroredStrategy(
	["/device:GPU:0", "/device:GPU:1", "/device:GPU:2", "/device:GPU:3", 
	"/device:GPU:4", "/device:GPU:5", "/device:GPU:6", "/device:GPU:7"])

config = tf.estimator.RunConfig(train_distribute=distribution, eval_distribute=distribution)
config = config.replace(save_summary_steps=100)

model = tf.estimator.Estimator(model_fn=model_fn, config=config,
	params=params, model_dir="checkpoints/" + exp_name + '/')




model.train(input_fn=train_input_fn, steps=10000)


result = model.evaluate(input_fn=test_input_fn)

print(result)
def serving_input_receiver_fn():
	inputs = {
	'feature': tf.placeholder(tf.float32, [None,240, 320, 3])}
	return tf.estimator.export.ServingInputReceiver(inputs, inputs)


model.export_savedmodel('models/' + exp_name + '/',
        serving_input_receiver_fn=serving_input_receiver_fn)
