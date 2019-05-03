import numpy as np

import os
import sys
import tensorflow as tf
import glob
import random
import cv2
from tensorflow.python.layers import base
import tensorflow.contrib.slim as slim




# tensorflow model directory
exported_path = sys.argv[1]
testset_path = sys.argv[2]

if len(sys.argv) != 3:
    print('one or more arguments missing! Please check arguments')
    sys.exit()

images = glob.glob(testset_path + '/*/*.jpg')
random.shuffle(images)

# we dont want to use gpu for the inference
config = tf.ConfigProto(device_count = {'GPU': 0})

num_images  = len(images)

def model_summary():
    model_vars = tf.trainable_variables()
    slim.model_analyzer.analyze_vars(model_vars, print_info=True)

def main():
    with tf.Session(config=config) as sess:
        tf.saved_model.loader.load(sess, [tf.saved_model.tag_constants.SERVING], exported_path)

        input_tensor=tf.get_default_graph().get_tensor_by_name("Placeholder:0")
        output_tensor=tf.get_default_graph().get_tensor_by_name("ArgMax:0")
        total = 0
        correct_pred = 0
        confusion_matrix = np.zeros((10,10))

        for img_path in images:
            total += 1
            label = int(img_path[img_path.rfind('/')-1])
            image = cv2.imread(img_path)
            image = cv2.resize(image, dsize=(320, 240), interpolation=cv2.INTER_CUBIC)
            image_np_expanded = np.expand_dims(image, axis=0)
            prediction = sess.run(output_tensor, feed_dict={input_tensor: image_np_expanded})

            confusion_matrix[label, prediction] += 1
            if total % 100 == 0:
                print(total,'/',num_images, 'completed')
            if label == int(prediction[0]):
                correct_pred += 1
        model_summary()
        print("confusion matrix")
        print('     c0  c1  c2  c3  c4  c5  c6  c7  c8  c9')
        for i in range(10):
            line = 'c' + str(i) + ' |'
            for j in range(10):
                line += '%3d ' % (confusion_matrix[i][j])
            print(line)
        print('correctly classified images:', correct_pred)
        print('total test images:', total)
        print("test accuracy:", correct_pred / total)




if __name__ == "__main__":
    main()

