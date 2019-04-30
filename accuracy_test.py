import numpy as np

import os
import sys
import tensorflow as tf
import glob
import random
from PIL import Image




# tensorflow model directory
exported_path = '/scratch/sakkas/models/model_1/1556419108'

images = glob.glob('dataset/hand-labeled/*/*.jpg')#['img_1.jpg', 'img_2.jpg', 'img_3.jpg', 'img_4.jpg']
random.shuffle(images)

# we dont want to use gpu for the inference
config = tf.ConfigProto(device_count = {'GPU': 0})

num_images  = len(images)

def main():
    with tf.Session(config=config) as sess:
        tf.saved_model.loader.load(sess, [tf.saved_model.tag_constants.SERVING], exported_path)

        input_tensor=tf.get_default_graph().get_tensor_by_name("Placeholder:0")
        output_tensor=tf.get_default_graph().get_tensor_by_name("ArgMax:0")
        total = 0
        correct_pred = 0
        for img_path in images:
            total += 1
            label = int(img_path[img_path.rfind('/')-1])
            image = np.array(Image.open(img_path))
            image_np_expanded = np.expand_dims(image, axis=0)
            prediction = sess.run(output_tensor, feed_dict={input_tensor: image_np_expanded})

            #print(img_path,'\t| prediction:', prediction[0], '\t | label:', label)
            if total % 100 == 0:
                print(total,'/',num_images, 'completed')
            if label == int(prediction[0]):
                correct_pred += 1
            if total ==  2000:
                break
        print("accuracy", correct_pred / total)




if __name__ == "__main__":
    main()

