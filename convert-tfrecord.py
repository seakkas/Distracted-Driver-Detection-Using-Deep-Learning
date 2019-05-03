# -*- coding: utf-8 -*-

import sys
import numpy as np
import tensorflow as tf
import glob
from random import shuffle
import cv2


train_image_dir = sys.argv[1]
test_image_dir = sys.argv[2]

# where to save tfrecord files
target_dir = sys.argv[3]

if len(sys.argv) != 4:
  print('one or more arguments missing! Please check arguments.')
  sys.exit()



def _int64_feature(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _bytes_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def get_image_binary(filename):
    image = cv2.imread(filename)
    image = cv2.resize(image, dsize=(320, 240), interpolation=cv2.INTER_CUBIC)

    # if gray scale wanted
    #image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    image = np.asarray(image, np.uint8)

    # this required if image is grayscale
    #image = np.expand_dims(image,axis=0)

    shape = np.array(image.shape, np.int32)
    return shape.tobytes(), image.tobytes() # convert image to raw data bytes in the array.

def write_to_tfrecord(images, tfrecord_file):
    writer = tf.python_io.TFRecordWriter(tfrecord_file)
    
    for img in images:
        shape, binary_image = get_image_binary(img)
        label = int(img[img.rfind('/')-1])
        # write label, shape, and image content to the TFRecord file
        example = tf.train.Example(features=tf.train.Features(feature={
                    'label': _int64_feature(label),
                    'shape': _bytes_feature(shape),
                    'image': _bytes_feature(binary_image)
                    }))
        writer.write(example.SerializeToString())
    writer.close()


# get all image files
# generated images are created by merging. 2 images are split 3:5
train_images = glob.glob(train_image_dir+'/*/*.jpg')

# shuffle images
shuffle(train_images)
  
#train_ratio = 0.85

#num_train_images = int(round(len(images) * train_ratio))
  
#train_images = images[:num_train_images]
#test_images = images[num_train_images:]

print('This operation will take some time')

write_to_tfrecord(train_images, target_dir + '/train.tfrecords')
print('training examples:',len(train_images))

test_images = glob.glob(test_image_dir + '/*/*.jpg')
shuffle(test_images)

write_to_tfrecord(test_images, target_dir + '/test.tfrecords')
print('test examples:',len(test_images))

