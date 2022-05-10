#%%
sys.path.append('./enlacement')
import imp
import sys
import numpy as np
import matplotlib.pyplot as plt
from skimage.morphology import convex_hull_image, area_opening, binary_erosion,binary_opening
import relation.surround as su
import tools as tl
import preprocess.Image_extract as ext
import enlacement.enlacement as en
imp.reload(tl)
imp.reload(su)
imp.reload(ext)
#%%
json_path = "../Qijia/ds0/ann/Tatish 1754 Andr¨¦ spatharocandidat . jpg copie A.jpg.json"
Image_dict = ext.Extract_image_mask(json_path)
print(Image_dict.keys())
# %%
# Extract differnt part of the image
left_flower = Image_dict["Fleuron_0"]
left_flower = tl.down_sample(left_flower)
#%%
hull = convex_hull_image(left_flower)
hull_boundary = hull-binary_erosion(hull)*1
plt.imshow(hull_boundary)
region = (hull_boundary+left_flower)
region = hull - region
region = region * (region>0)
plt.imshow(hull+left_flower,cmap = "gray")
plt.show()
plt.imshow(left_flower,cmap = "gray")
plt.show()
plt.imshow(region, cmap = "gray")
plt.show()
# %%
membership = su.surround(left_flower,region)
# %%
plt.imshow(membership, cmap = 'gray')
plt.show()
print(np.max(membership))
print(np.min(membership))
print(np.mean(membership[np.where(membership!=0)]))
# %%
json_path = "../Qijia/ds0/ann/Tatish 2805 AA 2016 copie.jpg.json"
Image_dict = ext.Extract_image_mask(json_path)
print(Image_dict.keys())
# %%
round = Image_dict["Grenetis_0"]
round = tl.down_sample(round)
plt.imshow(round,cmap ='gray')
plt.show()
np.max(round)
#%%
chull = convex_hull_image(round)
region = np.logical_xor(chull,round)
region = binary_opening(region)*1
plt.imshow(region,cmap = 'gray')
# %%
membership = su.surround(round,region,n_dir = 120)
plt.imshow(membership, cmap = 'gray')
plt.show()
print(np.max(membership))
print(np.min(membership))
print(np.mean(membership[np.where(membership!=0)]))

# %%
