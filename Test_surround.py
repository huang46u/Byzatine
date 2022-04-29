#%%
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import math
from skimage.measure import label,block_reduce
from sympy import comp, false
from skimage.morphology import convex_hull_image, binary_erosion,binary_dilation,binary_opening
import tools as tl
import imp
import enlacement.enlacement as en
imp.reload(tl)
#%%
im1 = np.array(Image.open("..\Qijia\ds0\masks_machine\Tatish 1754 Andr? spatharocandidat . jpg copie A.png"))
mask_machine = np.array(Image.open("..\Qijia\ds0\masks_machine\Tatish 1754 Andr? spatharocandidat . jpg copie A.png"))[:,:,0]
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
cross = (mask_machine==5)
marche = (mask_machine==2)
l_cross =  tl.down_sample(cross)
l_marche = tl.down_sample(marche)
#%%
chull = convex_hull_image(left_flower)
ero_chull = binary_erosion(chull)
plt.imshow(chull,cmap = 'gray')
plt.show()
plt.imshow(ero_chull,cmap = 'gray')
plt.show()
plt.imshow(left_flower,cmap = 'gray')
# %%
flower_boundary = left_flower- binary_erosion(left_flower)*1
hull_boundary = ero_chull-binary_erosion(ero_chull)*1
selem = np.array([[1,1],[1,1]])
db_Ndch = (flower_boundary - hull_boundary)
db_Ndch = (db_Ndch>0)*1
p = ero_chull-left_flower
p = binary_opening(p,selem=selem)*1

#%%
plt.imshow(chull)
plt.show()
plt.imshow(flower_boundary)
plt.show()
plt.imshow(hull_boundary)
plt.show()
plt.imshow(db_Ndch)
plt.show()
plt.imshow(p,cmap = 'gray')
plt.show()
#%%
# %%
membership = tl.surround(left_flower,p)
# %%
plt.imshow(membership, cmap = 'gray')
plt.show()
# %%
mask_machine = np.array(Image.open("..\Qijia\ds0\masks_machine\Tatish 2805 AA 2016 copie.png"))[:,:,0]
plt.imshow(mask_machine)
plt.show()

# %%
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
round = (labeled_image==1)*labeled_image
round = block_reduce(round,block_size=(16,16))*1
chull = convex_hull_image(round)
region = np.logical_xor(chull,round)
region = binary_opening(region)*1
plt.imshow(region,cmap = 'gray')
# %%
membership = tl.surround(round,region,bin = 120)
# %%
plt.imshow(round,cmap="gray")
plt.show()
plt.imshow(membership, cmap = 'gray')
plt.show()
# %%
tl.debug_surround(round,region)
#%%
