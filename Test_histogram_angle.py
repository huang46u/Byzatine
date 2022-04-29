#%%
from matplotlib import image
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import math
from skimage.measure import label,block_reduce
from sympy import comp, false
from skimage.morphology import convex_hull_image, binary_erosion,binary_dilation,binary_opening
import tools as tl
import imp
imp.reload(tl)
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
cross = (mask_machine==5)
marche = (mask_machine==2)
l_cross = tl.down_sample(cross)
l_marche = tl.down_sample(marche)
# %%
tl.demo_histogram_force(l_cross,l_marche)
# %%
tl.plot_morpho(l_cross,l_marche)
# %%
tl.Histogram_Angle(l_cross,l_marche)
# %%
tl.plot_morpho(cross,left_flower)
# %%
tl.Histogram_Angle(cross,left_flower)
   

# %%
