#%%
sys.path.append('./enlacement')
import imp
import sys
from turtle import left
import numpy as np
import matplotlib.pyplot as plt
from skimage.morphology import convex_hull_image, area_opening, binary_erosion,binary_opening
import relation.surround as su
import tools as tl
import file_processing.Image_extract as ext
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
region = su.candidate_region(left_flower)
plt.imshow(region)
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
Marche1 = Image_dict['Marche_0']
Marche2 = Image_dict['Marche_1']
Marche3 = Image_dict['Marche_2']
Marche = Marche1 + Marche2 +Marche3
Croix = Image_dict['Croix_0']
round = Image_dict["Grenetis_0"]
round = tl.down_sample(round)
croix = tl.down_sample(Croix)
marche = tl.down_sample(Marche)
plt.xlabel("Grenetis")
plt.imshow(round,cmap ='gray')
plt.show()
plt.imshow(Marche, cmap = 'gray')
plt.show()
plt.xlabel("Marche")
plt.imshow(marche, cmap = "gray")
plt.show()
plt.xlabel("Croix")
plt.imshow(croix, cmap = "gray")

#%%
region = su.candidate_region(round)
plt.imshow(region)
#%%
region = binary_opening(region)*1
plt.imshow(region,cmap = 'gray')
# %%
membership = su.surround(round,region,n_dir = 20)
# %%
plt.imshow(membership, cmap = 'gray')
plt.show()
print(np.max(membership))
print(np.min(membership))
print(np.mean(membership[np.where(membership!=0)]))

# %%
img = np.minimum(membership, marche)
img += np.minimum(membership, croix)
plt.xlabel("Relation Surround")
plt.imshow(img, cmap = "gray")

# %%
plt.xlabel("Fuzzy landscape")
plt.imshow(membership, cmap = 'gray')
plt.show()

# %%
val= img[np.where(img!=0)]
print(np.max(val))
print(np.min(val))
print(np.mean(val))

# %%
