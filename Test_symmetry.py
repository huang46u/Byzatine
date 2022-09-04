#%%
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import tools as tl
import plot

from skimage import transform
import file_processing.Image_extract as ext
import imp
import relation.symmetry as sm
imp.reload(sm)
imp.reload(tl)
imp.reload(plot)
# %%
#Symmetry example 1
#original image
image = plt.imread("../Zacos-genevre/Zacos-Geneve/masks_machine/cdn_2004_0355_A.png")
plt.imshow(image)
# extract hand mask
hand = ext.Extract_image_part("D:/M2/Stage/Byzatine/Zacos-genevre/Zacos-Geneve/ann/cdn_2004_0355_A.jpg.json",'classTitle','Part_hand')
hand_image = hand['Part_hand0'] + hand['Part_hand1']
plt.imshow(hand_image,cmap="gray")
init = np.array([0.5*np.pi, 0])
diff = sm.symmetry(hand_image,init)
# %%
#Symmetry example 2
image = plt.imread("../Zacos-genevre/Zacos-Geneve/masks_human/cdn_2004_0322_A.png")
plt.imshow(image)
wings = ext.Extract_image_part("D:/M2/Stage/Byzatine/Zacos-genevre/Zacos-Geneve/ann/cdn_2004_0322_A.jpg.json",'classTitle','Part_Wings')
wings_image = wings['Part_Wings0'] + wings['Part_Wings1']
plt.imshow(wings_image)
init = np.array([0.5*np.pi, 0])
diff = sm.symmetry(wings_image,init)

# %%
#Compute missing part based on symmetry relation
#Example 1
image = plt.imread("../Qijia/ds0/masks_human/Tatish 2363 patrice stratège  copie 2 A.png")
plt.imshow(image)
flower = ext.Extract_image_part("../Qijia/ds0/ann/Tatish 2363 patrice stratège  copie 2 A.jpg.json",'classTitle','Fleuron')
flower_image = flower["Fleuron0"]
plt.imshow(flower_image,cmap="gray")
init = np.array([0.5*np.pi, 30])
diff = sm.symmetry(flower_image,init)
sm.comput_missing_part(flower_image,diff,1024)
# %%
#Example 2
cross = ext.Extract_image_part("../Qijia/ds0/ann/Tatish 2451 Michel  copie A.jpg.json",'classTitle',"Croix_recroisetee")
cross_image = cross["Croix_recroisetee0"]
w,h = cross_image.shape
cross_image = transform.resize(cross_image,(w,w))
plt.imshow(cross_image,cmap="gray")
init = np.array([0.5*np.pi, -60])
diff = sm.symmetry(cross_image,init)
sm.comput_missing_part(cross_image,diff,2048)
# %%
#Example 3
flower = ext.Extract_image_part("../Qijia/ds0/ann/Tatish 2780 AA 2016 copie 2.jpg.json",'classTitle','Fleuron')
flower_image = flower["Fleuron0"]
w,h = flower_image.shape
flower_image = transform.resize(flower_image,(w,w))
plt.imshow(flower_image,cmap="gray")
init = np.array([0.5*np.pi, -80])
diff = sm.symmetry(flower_image,init)
sm.comput_missing_part(flower_image,diff,10000)
# %%


