from glob import glob
import os
import matplotlib.pyplot as plt

from utils import *


def store_single(filename, height , width, col_num, sel):
    img = scipy.misc.imread(filename).astype(np.float)
    name, extension = filename.split('.')
    col, row = sel % col_num, sel / col_num
    sel_image = img[row*height:(row+1)*height, col*width:(col+1)*width, :]
    scipy.misc.imsave(name + '_single.' + extension, sel_image)

filename = '5800.png'
height , width, col_num = 256, 512, 4
sel = 5
store_single(filename, height , width, col_num, sel)