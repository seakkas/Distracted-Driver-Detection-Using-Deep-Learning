import cv2
import glob
import os


#https://gist.github.com/jdhao/f8422980355301ba30b6774f610484f2

def pad_resize_save(im_paths, out_dir):
  desired_size = 299
  for im_path in im_paths:
    im_class = im_path[im_path.rfind("/")-1]
    im_name = im_path[im_path.rfind("/")+1:]

    im = cv2.imread(im_path)
    old_size = im.shape[:2] # old_size is in (height, width) format
    ratio = float(desired_size)/max(old_size)
    new_size = tuple([int(x*ratio) for x in old_size])
  
    # new_size should be in (width, height) format
    im = cv2.resize(im, (new_size[1], new_size[0])) 

    delta_w = desired_size - new_size[1]
    delta_h = desired_size - new_size[0]
    top, bottom = delta_h//2, delta_h-(delta_h//2)
    left, right = delta_w//2, delta_w-(delta_w//2)
    color = [255, 204, 203]
    new_im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    os.makedirs(out_dir + '/c' + im_class +'/', exist_ok=True)
    cv2.imwrite(out_dir + '/c' + im_class +'/' + im_name, new_im)


train_im_paths = glob.glob('dataset/org/train/c*/*.jpg')
test_im_paths = glob.glob('dataset/org/hand-labeled/c*/*.jpg')


pad_resize_save(train_im_paths, 'dataset/pad_resize/train')
pad_resize_save(test_im_paths, 'dataset/pad_resize/hand-labeled')