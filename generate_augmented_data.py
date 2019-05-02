import Augmentor
import glob
import os
import shutil
import time



def augment_data(im_path):
  num_images = len(glob.glob(im_path + '/*.jpg'))
  p = Augmentor.Pipeline(im_path)
  #p.flip_left_right(probability=0.3)
  p.rotate(probability=0.4, max_left_rotation=15, max_right_rotation=15)
  p.random_brightness(probability=0.4, min_factor=0.7, max_factor=1.3)
  p.shear(probability=0.4, max_shear_left=15, max_shear_right=15)
  p.random_distortion(probability=0.4, grid_width=8, grid_height=8, magnitude=5)
  p.zoom(probability=0.4, min_factor=1.1, max_factor=1.5)
  p.sample(num_images * 2)



def copy_original_images(im_path, target_dir):
  images = glob.glob(im_path + '/*.jpg')
  for img in images:
    shutil.copy2(img, target_dir)


#train_im_paths = glob.glob('dataset/pad_resize/train/c0/*')[10]
for i in range(10):
  if os.path.exists('dataset/augmented/train/c' + str(i)):
    shutil.rmtree('dataset/augmented/train/c' + str(i))
  augment_data ('dataset/org/train/c' + str(i))
  shutil.move('dataset/org/train/c' + str(i) +'/output', 'dataset/augmented/train/c' + str(i))
  copy_original_images('dataset/org/train/c' + str(i), 'dataset/augmented/train/c' + str(i))