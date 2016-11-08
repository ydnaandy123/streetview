from glob import glob
import Image
import os
import scipy.misc
import numpy as np
import cv2
import matplotlib.pyplot as plt

from utils import *


CITYSCAPES_dir = "/home/andy/dataset/CITYSCAPES/leftImg8bit_trainvaltest/leftImg8bit/train"
#CITYSCAPES_dir = "/home/andy/dataset/CITYSCAPES/gtFine_trainvaltest/gtFine/train"
#training_dir = "/home/andy/dataset/CITYSCAPES/CITYSCAPES_crop_random_mask"
#trainings = glob(os.path.join(training_dir, "*.png"))
#mask_dir = "/home/andy/dataset/CITYSCAPES/CITYSCAPES_crop_random"
#masks = glob(os.path.join(mask_dir, "*.png"))

'''
for index, filePath in enumerate(trainings):
    print ('%d/%d' % (index, len(trainings)))
    img = scipy.misc.imread(filePath).astype(np.float)
    plt.imshow(img)

    scipy.misc.imsave('/home/andy/dataset/CITYSCAPES/CITYSCAPES_crop_random_mask/' + filePath.split('/')[-1], mask)
    break
'''

'''
"""
Create masks
"""
for folder in os.listdir(CITYSCAPES_dir):
    path = os.path.join(CITYSCAPES_dir, folder, "*_color.png")
    data.extend(glob(path))

data_length = len(data)

for index, filePath in enumerate(data):
    print ('%d/%d' % (index, len(data)))
    img = scipy.misc.imread(filePath).astype(np.float)
    img = scipy.misc.imresize(img, 0.25, interp='bilinear', mode=None)


    mask = np.ones((img.shape[0], img.shape[1]))

    idx_person = np.where(np.all(img == [220, 20, 60, 255], axis=-1))
    idx_rider = np.where(np.all(img == [255, 0, 0, 255], axis=-1))
    idx_void = np.where(np.all(img == [0, 0, 0, 255], axis=-1))

    indices = np.concatenate((idx_person, idx_rider, idx_void), axis=1)
    #mask[indices[0], indices[1], :] = (0, 0, 0, 255)
    mask[indices[0], indices[1]] = 0
    #scipy.misc.imsave('/home/andy/dataset/CITYSCAPES/CITYSCAPES_crop_random/' + filePath.split('/')[-1],
    #                  img[offs_h[index]:offs_h_end[index], offs_w[index]:offs_w_end[index] :])
    scipy.misc.imsave('/home/andy/dataset/CITYSCAPES/CITYSCAPES_crop_random_mask/' + filePath.split('/')[-1], mask)
'''

"""
random crop a image [256, 512]
in the resize (x0.5) original image [512, 1024]
"""

data = []
for folder in os.listdir(CITYSCAPES_dir):
    #path = os.path.join(CITYSCAPES_dir, folder, "*.png")
    path = os.path.join(CITYSCAPES_dir, folder, "*.png")
    data.extend(glob(path))

data_length = len(data)
# high < 256 because want to cut the bottom
offs_h = np.random.randint(low=0, high=200, size=data_length)
offs_h_end = offs_h + 256
offs_w = np.random.randint(low=0, high=512, size=data_length)
offs_w_end = offs_w + 512
print offs_h, offs_h_end
#print offs_h

for index, filePath in enumerate(data):
    print ('%d/%d' % (index, len(data)))

    img = scipy.misc.imread(filePath).astype(np.float)
    img = scipy.misc.imresize(img, 0.25, interp='bilinear', mode=None)
    #scipy.misc.imsave('/home/andy/dataset/CITYSCAPES/CITYSCAPES_crop_random/' + filePath.split('/')[-1],
    #                  img[offs_h[index]:offs_h_end[index], offs_w[index]:offs_w_end[index] :])
    scipy.misc.imsave('/home/andy/dataset/CITYSCAPES/CITYSCAPES_crop_bottom/' + filePath.split('/')[-1], img[0:200, :, :])
    #break



def resize_and_crop(img_path, modified_path, size, crop_type='top'):
    """
    Resize and crop an image to fit the specified size.

    args:
    img_path: path for the image to resize.
    modified_path: path to store the modified image.
    size: `(width, height)` tuple.
    crop_type: can be 'top', 'middle' or 'bottom', depending on this
    value, the image will cropped getting the 'top/left', 'middle' or
    'bottom/right' of the image to fit the size.
    raises:
    Exception: if can not open the file in img_path of there is problems
    to save the image.
    ValueError: if an invalid `crop_type` is provided.
    """
    # If height is higher we resize vertically, if not we resize horizontally
    img = Image.open(img_path)
    # Get current and desired ratio for the images
    img_ratio = img.size[0] / float(img.size[1])
    ratio = size[0] / float(size[1])
    #The image is scaled/cropped vertically or horizontally depending on the ratio
    if ratio > img_ratio:
        img = img.resize((size[0], int(round(size[0] * img.size[1] / img.size[0]))),
            Image.ANTIALIAS)
        # Crop in the top, middle or bottom
        if crop_type == 'top':
            box = (0, 0, img.size[0], size[1])
        elif crop_type == 'middle':
            box = (0, int(round((img.size[1] - size[1]) / 2)), img.size[0],
                int(round((img.size[1] + size[1]) / 2)))
        elif crop_type == 'bottom':
            box = (0, img.size[1] - size[1], img.size[0], img.size[1])
        else :
            raise ValueError('ERROR: invalid value for crop_type')
        img = img.crop(box)
    elif ratio < img_ratio:
        img = img.resize((int(round(size[1] * img.size[0] / img.size[1])), size[1]),
            Image.ANTIALIAS)
        # Crop in the top, middle or bottom
        if crop_type == 'top':
            box = (0, 0, size[0], img.size[1])
        elif crop_type == 'middle':
            box = (int(round((img.size[0] - size[0]) / 2)), 0,
                int(round((img.size[0] + size[0]) / 2)), img.size[1])
        elif crop_type == 'bottom':
            box = (img.size[0] - size[0], 0, img.size[0], img.size[1])
        else :
            raise ValueError('ERROR: invalid value for crop_type')
        img = img.crop(box)
    else :
        img = img.resize((size[0], size[1]),
            Image.ANTIALIAS)
    # If the scale is the same, we do not need to crop
    img.save(modified_path)
    

