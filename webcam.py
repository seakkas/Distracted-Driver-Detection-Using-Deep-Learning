import numpy as np

import os
import sys
import tensorflow as tf
import glob
from PIL import Image
import cv2




exported_path = '1556335034'

# we dont want to use gpu for the inference
config = tf.ConfigProto(device_count = {'GPU': 0})

camera = cv2.VideoCapture(0)

labels = {
    0: 'safe driving',
    1 : 'texting - right',
    2 : 'talking on the phone - right',
    3 : 'texting - left',
    4 : 'talking on the phone - left',
    5 : 'operating the radio',
    6 : 'drinking',
    7 : 'reaching behind',
    8 : 'hair and makeup',
    9 : 'talking to passenger'
    }







def grabVideoFeed():
    grabbed, frame = camera.read()
    return frame if grabbed else None

def main():
    with tf.Session(config=config) as sess:
        tf.saved_model.loader.load(sess, [tf.saved_model.tag_constants.SERVING], exported_path)

        input_tensor=tf.get_default_graph().get_tensor_by_name("Placeholder:0")
        output_tensor=tf.get_default_graph().get_tensor_by_name("ArgMax:0")

        while True:
            frame = grabVideoFeed()

            if frame is None:
                raise SystemError('Issue grabbing the frame')

            frame = cv2.resize(frame, (640, 480), interpolation=cv2.INTER_CUBIC)
            image_np_expanded = np.expand_dims(frame, axis=0)

            prediction = sess.run(output_tensor, feed_dict={input_tensor: image_np_expanded})

            text = labels[int(prediction[0])]
            font_scale = 1.0
            font = cv2.FONT_HERSHEY_SIMPLEX
            # set the rectangle background to white
            rectangle_bgr = (255, 255, 255)
            # make a black image
            
            # get the width and height of the text box
            (text_width, text_height) = cv2.getTextSize(text, font, fontScale=font_scale, thickness=2)[0]
            # set the text start position
            text_offset_x = 10
            text_offset_y = frame.shape[0] - 25
            # make the coords of the box with a small padding of two pixels
            box_coords = ((text_offset_x, text_offset_y+5), (text_offset_x + text_width - 2, text_offset_y - text_height-2))
            cv2.rectangle(frame, box_coords[0], box_coords[1], rectangle_bgr, cv2.FILLED)
            cv2.putText(frame, text, (text_offset_x, text_offset_y), font, fontScale=font_scale, color=(0, 0, 0), thickness=1)
            #cv2.putText(frame, 'hello', (480, 30), fontScale=1, fontFace=cv2.FONT_HERSHEY_SIMPLEX, color=(255, 255, 255))
            cv2.imshow('main window', frame)



            if cv2.waitKey(1) & 0xFF == ord('q'):
                break




if __name__ == "__main__":
    main()

