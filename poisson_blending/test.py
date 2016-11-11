from scipy import misc
import os
face = misc.imread('nr_001_img_000000025_c0_re.png')
misc.imsave(os.path.join('source_mask{}.png'.format(1)), face)

import matplotlib.pyplot as plt
plt.imshow(face)
plt.show()