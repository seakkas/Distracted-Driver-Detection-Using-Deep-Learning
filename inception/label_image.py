# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse

import numpy as np
import tensorflow as tf
import glob

def load_graph(model_file):
  graph = tf.Graph()
  graph_def = tf.GraphDef()

  with open(model_file, "rb") as f:
    graph_def.ParseFromString(f.read())
  with graph.as_default():
    tf.import_graph_def(graph_def)

  return graph


def read_tensor_from_image_file(file_name):
  input_height=299
  input_width=299
  input_mean=0
  input_std=255
  #input_name = "file_reader"
  #output_name = "normalized"
  filename = tf.cast(file_name,tf.string)

  file_reader = tf.read_file(filename)
  image_reader = tf.image.decode_jpeg(
  	file_reader, channels=3, name="jpeg_reader")
  float_caster = tf.cast(image_reader, tf.float32)
  dims_expander = tf.expand_dims(float_caster, 0)
  resized = tf.image.resize_bilinear(dims_expander, [input_height, input_width])
  normalized = tf.divide(tf.subtract(resized, [input_mean]), [input_std])
  return normalized


def load_labels(label_file):
  label = []
  proto_as_ascii_lines = tf.gfile.GFile(label_file).readlines()
  for l in proto_as_ascii_lines:
    label.append(l.rstrip())
  return label


if __name__ == "__main__":
  file_name = "tensorflow/examples/label_image/data/grace_hopper.jpg"
  model_file = \
    "tensorflow/examples/label_image/data/inception_v3_2016_08_28_frozen.pb"
  label_file = "labels.txt"
  input_height = 299
  input_width = 299
  input_mean = 0
  input_std = 255
  input_layer = "InputImage"
  output_layer = "final_result"

  parser = argparse.ArgumentParser()
  parser.add_argument("--image_dir", help="image directory")
  parser.add_argument("--graph", help="graph/model to be executed")
  parser.add_argument("--labels", help="name of file containing labels")
  parser.add_argument("--input_height", type=int, help="input height")
  parser.add_argument("--input_width", type=int, help="input width")
  parser.add_argument("--input_mean", type=int, help="input mean")
  parser.add_argument("--input_std", type=int, help="input std")
  parser.add_argument("--input_layer", help="name of input layer")
  parser.add_argument("--output_layer", help="name of output layer")
  args = parser.parse_args()

  if args.graph:
    model_file = args.graph
  if args.image_dir:
    file_names = glob.glob(args.image_dir + '/c*/*.jpg' )
  if args.labels:
    label_file = args.labels
  if args.input_height:
    input_height = args.input_height
  if args.input_width:
    input_width = args.input_width
  if args.input_mean:
    input_mean = args.input_mean
  if args.input_std:
    input_std = args.input_std
  if args.input_layer:
    input_layer = args.input_layer
  if args.output_layer:
    output_layer = args.output_layer

  graph = load_graph(model_file)

  input_name = "import/" + input_layer
  output_name = "import/" + output_layer
  input_operation = graph.get_operation_by_name(input_name)
  output_operation = graph.get_operation_by_name(output_name)
  total_image = 0
  correct_prediction = 0
  labels = load_labels(label_file)

  confusion_matrix = np.zeros((10,10))
  
  config = tf.ConfigProto(device_count = {'GPU': 0})


  with tf.Session(graph=graph, config=config) as sess:
    output_op = output_operation.outputs[0]
    input_op = input_operation.outputs[0]
    dataset = tf.data.Dataset.from_tensor_slices(file_names)
    dataset = dataset.map(read_tensor_from_image_file, num_parallel_calls=4)
    #dataset = dataset.batch(1)
    iterator = dataset.make_one_shot_iterator()
    next_element = iterator.get_next()

    for file_name in file_names: 
      #t = read_tensor_from_image_file(
       # file_name,
        #input_height=input_height,
      	#input_width=input_width,
      	#input_mean=input_mean,
      	#input_std=input_std)

      img = sess.run(next_element)
      results = sess.run(output_op, {input_op: img})
      results = np.squeeze(results)
      top_k = results.argsort()[-5:][::-1]
      real_class = int(file_name[file_name.rfind('/')-1])
      predicted_class = int(labels[top_k[0]][1])
      confusion_matrix[real_class, predicted_class] += 1
      if predicted_class == real_class:
      	correct_prediction += 1
      total_image += 1
      if total_image % 100 == 0:
      	print(total_image, ' images predicted')

    print("confusion matrix")
    print('     c0  c1  c2  c3  c4  c5  c6  c7  c8  c9')
    for i in range(10):
      line = 'c' + str(i) + ' |'
      for j in range(10):
        line += '%3d ' % (confusion_matrix[i][j])
      print(line)

    print('correctly classified images:', correct_prediction)
    print('total test images:', total_image)
    print('test accuracy:', correct_prediction / total_image)


  #labels = load_labels(label_file)
  #for i in top_k:
  #  print(labels[i], results[i])
