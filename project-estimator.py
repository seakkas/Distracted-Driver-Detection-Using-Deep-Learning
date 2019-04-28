# -*- coding: utf-8 -*-

from __future__ import absolute_import, division, print_function

import tensorflow as tf
import numpy as np
import time
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


def read_dataset(file_names, mode, batch_size=64, prefetch=8):
    dataset = tf.data.TFRecordDataset(file_names)
    dataset = dataset.map(_parse_function, num_parallel_calls=48)
    if mode == tf.estimator.ModeKeys.TRAIN:
    	dataset = dataset.repeat()
    #	dataset = dataset.shuffle(512)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(buffer_size=prefetch)
    return dataset



def model_fn(features, labels, mode, params):

	inputs = tf.reshape(features['feature'], [-1, 240, 320, 1])
    # we use xavier initializer to initialize variables
	xav_init = tf.contrib.layers.xavier_initializer()
	
	conv1 = tf.layers.conv2d(inputs=inputs, filters=32, kernel_size=[5, 5], padding="same", activation=tf.nn.relu, kernel_initializer=xav_init)
	
	pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[3, 3], strides=3)

	# Convolutional Layer #2 and Pooling Layer #2
	conv2 = tf.layers.conv2d(inputs=pool1, filters=64, kernel_size=[5, 5], padding="same", activation=tf.nn.relu, kernel_initializer=xav_init)
	pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[3, 3], strides=3)
	
	conv3 = tf.layers.conv2d(inputs=pool2, filters=32, kernel_size=[5, 5], padding="same", activation=tf.nn.relu, kernel_initializer=xav_init)
	pool3 = tf.layers.max_pooling2d(inputs=conv3, pool_size=[2, 2], strides=2)

	conv4 = tf.layers.conv2d(inputs=pool3, filters=16, kernel_size=[5, 5], padding="same", activation=tf.nn.relu, kernel_initializer=xav_init)
	pool4 = tf.layers.max_pooling2d(inputs=conv4, pool_size=[2, 2], strides=2)

	# Dense Layer
	flat = tf.layers.flatten(pool4)
	training = (mode == tf.estimator.ModeKeys.TRAIN)
	dense1 = tf.layers.dense(inputs=flat, units=512, activation=tf.nn.relu, kernel_initializer=xav_init)
	dropout1 = tf.layers.dropout(inputs=dense1, rate=0.4, training=training)
	dense2 = tf.layers.dense(inputs=dropout1, units=128, activation=tf.nn.relu, kernel_initializer=xav_init)
	dropout2 = tf.layers.dropout(inputs=dense2, rate=0.4, training=training)

	# Logits Layer
	logits = tf.layers.dense(inputs=dropout2, units=10, kernel_initializer=xav_init)

	# Softmax output of the neural network.
	y_pred = tf.nn.softmax(logits=logits)

    # Classification output of the neural network.
	y_pred_cls = tf.argmax(y_pred, axis=1)

	if mode == tf.estimator.ModeKeys.PREDICT:
        # If the estimator is supposed to be in prediction-mode
        # then use the predicted class-number that is output by
        # the neural network. Optimization etc. is not needed.
		spec = tf.estimator.EstimatorSpec(mode=mode, predictions=y_pred_cls)
	else:
    	# Otherwise the estimator is supposed to be in either
    	# training or evaluation-mode. Note that the loss-function
    	# is also required in Evaluation mode.

    	# Define the loss-function to be optimized, by first
    	# calculating the cross-entropy between the output of
    	# the neural network and the true labels for the input data.
    	# This gives the cross-entropy for each image in the batch.
		cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(labels=labels, logits=logits)

		# Reduce the cross-entropy batch-tensor to a single number
		# which can be used in optimization of the neural network.
		loss = tf.reduce_mean(cross_entropy)
		# Define the optimizer for improving the neural network.
		optimizer = tf.train.AdamOptimizer(params["learning_rate"])

		# Get the TensorFlow op for doing a single optimization step.
		train_op = optimizer.minimize(loss=loss, global_step=tf.train.get_global_step())

    	# Define the evaluation metrics,
    	# in this case the classification accuracy.
		accuracy = tf.metrics.accuracy(labels=tf.argmax(labels, axis=1), predictions=y_pred_cls, name='acc_op')
		metrics = {"accuracy": accuracy}
		tf.summary.scalar('accuracy', accuracy[1])
		#tf.summary.scalar('myloss', loss)
		#train_hook_list= []
		#train_tensors_log = {'accuracy': accuracy[1], 'loss': loss, 'global_step': tf.train.get_global_step()}
		#train_hook_list.append(tf.train.LoggingTensorHook(tensors=train_tensors_log, every_n_iter=5))

    	# Wrap all of this in an EstimatorSpec.
		spec = tf.estimator.EstimatorSpec(mode=mode,loss=loss,train_op=train_op,
                eval_metric_ops=metrics)

	return spec





train_input_fn = lambda: read_dataset(file_names='train.tfrecords',
        mode=tf.estimator.ModeKeys.TRAIN, batch_size=32)
test_input_fn = lambda: read_dataset(file_names='test.tfrecords', mode=tf.estimator.ModeKeys.EVAL,
        batch_size=32)

params = {"learning_rate": 1e-4}


# to use multiple GPU
distribution = tf.contrib.distribute.MirroredStrategy(
	["/device:GPU:0", "/device:GPU:1", "/device:GPU:2", "/device:GPU:3", 
	"/device:GPU:4", "/device:GPU:5", "/device:GPU:6", "/device:GPU:7"])
config = tf.estimator.RunConfig(train_distribute=distribution, eval_distribute=distribution)
config = config.replace(save_summary_steps=5)


model = tf.estimator.Estimator(model_fn=model_fn, config=config,
	params=params, model_dir="checkpoints/model_1/")




model.train(input_fn=train_input_fn, steps=2000)


result = model.evaluate(input_fn=test_input_fn)

print(result)
def serving_input_receiver_fn():
	inputs = {
	'feature': tf.placeholder(tf.float32, [None,240, 480, 3])}
	return tf.estimator.export.ServingInputReceiver(inputs, inputs)


model.export_savedmodel('models/model_1/',
        serving_input_receiver_fn=serving_input_receiver_fn)
