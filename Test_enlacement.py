#%%
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import math
from skimage.measure import label,block_reduce
from sympy import comp, false
from skimage.morphology import convex_hull_image, binary_erosion,binary_dilation,binary_opening
from torch import le
from tools import surround
import tools as tl
import seaborn as sns
imp.reload(tl)
import imp
import sys
sys.path.append('./enlacement')
import enlacement.enlacement as en


#%%
im1 = np.array(Image.open("..\Qijia\ds0\masks_machine\Tatish 1754 Andr¨¦ spatharocandidat . jpg copie A.png"))
mask_machine = np.array(Image.open("..\Qijia\ds0\masks_machine\Tatish 1754 Andr¨¦ spatharocandidat . jpg copie A.png"))[:,:,0]
plt.imshow(mask_machine)
plt.show()
plt.imshow(im1)
#%%
labeled_image = label(mask_machine)
print(tl.unique_value(labeled_image))
plt.imshow((labeled_image==1)*labeled_image)
plt.show()
plt.imshow((labeled_image==2)*labeled_image)
plt.show()
plt.imshow((labeled_image==3)*labeled_image)
plt.show()
plt.imshow((labeled_image==4)*labeled_image)
plt.show()
# %%
# Extract differnt part of the image
left_flower = (labeled_image==1)
cross = (labeled_image==2)
cross =  tl.down_sample(cross)
left_flower = tl.down_sample(left_flower)
#%%
#%%
cross = (mask_machine==5)
marche = (mask_machine==2)
l_cross =  tl.down_sample(cross)
l_marche = tl.down_sample(marche)
#%%
plt.imshow(cross+left_flower)
plt.show()

#%%
i_ab,e_ab, e_ba = en.interlacement(cross,left_flower,n_jobs = None)
#%%
step = math.pi/180
bins = np.arange(0,math.pi+step,step)
width = np.diff(bins)
center = (bins[:-1] + bins[1:]) / 2
plt.bar(center, e_ab, align='center', width=width)
plt.show()
step = math.pi/180
bins = np.arange(0,math.pi+step,step)
width = np.diff(bins)
center = (bins[:-1] + bins[1:]) / 2
plt.bar(center, e_ba, align='center', width=width)
plt.show()
step = math.pi/180
bins = np.arange(0,math.pi+step,step)
width = np.diff(bins)
center = (bins[:-1] + bins[1:]) / 2
plt.bar(center, i_ab, align='center', width=width)
plt.show()
# %%
en.surrounding(e_ab)
# %%

# %%
