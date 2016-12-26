from glob import glob
import os
import matplotlib.pyplot as plt

from utils import *

CITYSCAPES_dir = "/home/andy/dataset/CITYSCAPES/leftImg8bit_trainvaltest/leftImg8bit/train"


"""
filename = '5800.png'
height , width, col_num = 256, 512, 4
sel = 5
store_single(filename, height , width, col_num, sel)
"""


def store_single(filename, height , width, col_num, sel):
    img = scipy.misc.imread(filename).astype(np.float)
    name, extension = filename.split('.')
    col, row = sel % col_num, sel / col_num
    sel_image = img[row*height:(row+1)*height, col*width:(col+1)*width, :]
    scipy.misc.imsave(name + '_' + str(sel) + '.' + extension, sel_image)



filename = '/home/andy/Github/streetview/DCGAN_streetview/completions/0800.png'
height , width, col_num = 192, 512, 4
sel = 12
store_single(filename, height , width, col_num, sel)

"""
crop_images('/mnt/data/andy/dataset/CITYSCAPES/gtFine_trainvaltest/gtFine/train', is_mask=True)
"""
def crop_images(dataset_dir, is_mask=False):
    data = []
    for folder in os.listdir(dataset_dir):
        path = os.path.join(dataset_dir, folder, "*.jpg")
        data.extend(glob(path))

    for index, filePath in enumerate(data):
        print ('{}/{}'.format(index, len(data)))

        img = scipy.misc.imread(filePath).astype(np.uint8)
        #img = scipy.misc.imresize(img, 0.25, interp='bilinear', mode=None)
        if is_mask:
            mask = np.ones((img.shape[0], img.shape[1]), dtype=np.uint8) * 255

            idx_person = np.where(np.all(img == [220, 20, 60, 255], axis=-1))
            #idx_rider = np.where(np.all(img == [255, 0, 0, 255], axis=-1))
            #idx_void = np.where(np.all(img == [0, 0, 0, 255], axis=-1))

            #indices = np.concatenate((idx_person, idx_rider, idx_void), axis=1)
            indices = idx_person
            # mask[indices[0], indices[1], :] = (0, 0, 0, 255)
            mask[indices[0], indices[1]] = 0
            mask = np.reshape(mask, (256, 512))

        #scipy.misc.imsave('/home/andy/dataset/CITYSCAPES/CITYSCAPES_crop_random/' + filePath.split('/')[-1],
        #                  img[offs_h[index]:offs_h_end[index], offs_w[index]:offs_w_end[index] :])
        scipy.misc.imsave('/mnt/data/andy/dataset/indoor_my/{}_{}'.format(filePath.split('/')[-2], filePath.split('/')[-1]),
                          img)
        #break



"""
crop_images_same_dir('/mnt/data/andy/dataset/CITYSCAPES/CITYSCAPES_crop_bottom')
"""
def crop_images_same_dir(data_set_dir):
    data = glob(os.path.join(data_set_dir, "*.png"))

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
        #img = scipy.misc.imresize(img, 0.25, interp='bilinear', mode=None)
        #scipy.misc.imsave('/home/andy/dataset/CITYSCAPES/CITYSCAPES_crop_random/' + filePath.split('/')[-1],
        #                  img[offs_h[index]:offs_h_end[index], offs_w[index]:offs_w_end[index] :])
        scipy.misc.imsave('/home/andy/dataset/CITYSCAPES/CITYSCAPES_crop_bottom_192/' + filePath.split('/')[-1],
                          img[0:192, :, :])
        #break

