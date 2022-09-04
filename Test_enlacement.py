#%%
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import math
from skimage.measure import label
import tools as tl
import file_processing.Image_extract as ext
import imp
import sys
sys.path.append('./enlacement')
import enlacement.enlacement as en
imp.reload(tl)
#%%
json_path = "../Qijia/ds0/ann/Tatish 1754 Andr¨¦ spatharocandidat . jpg copie A.jpg.json"
Image_dict = ext.Extract_image_mask(json_path)
print(Image_dict.keys())
#%%
left_flower = Image_dict["Fleuron_1"]
cross = Image_dict["Croix_recroisetee_1"]
cross =  tl.down_sample(cross)
left_flower = tl.down_sample(left_flower)

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
