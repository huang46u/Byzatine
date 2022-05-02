#%%
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import tools as tl
import plot
from preprocess import *
from skimage.morphology import area_opening
from skimage import transform
import preprocess.Image_extract as ext
import imp
import cv2
import sys
imp.reload(tl)
imp.reload(plot)

def symmetry(image, param_init):
    print("init value: ",tl.symetry_measure(param_init,image))
    symm_original,line_original = tl.symmetry_plot_data(image,param_init)
    plot.plot_sym_plan(image, symm_original,line_original)
    opt = tl.downhill_simplex(image,param_init)
    print(opt)
    symm_optimized, line_optimized = tl.symmetry_plot_data(image,opt)
    plot.plot_sym_plan(image, symm_optimized, line_optimized)
    diff = symm_optimized - image
    return diff

def comput_missing_part(image,diff,threshold ):
    pos_diff = diff * (diff>0)
    plt.figure(figsize=(10,10))
    plt.subplot(1,2,1)
    plt.xlabel("positive_difference")
    plt.imshow(diff* (diff>0),cmap = 'gray')
    plt.subplot(1,2,2)
    plt.xlabel("closing")
    last_part = area_opening(pos_diff, area_threshold=threshold)
    plt.imshow(last_part, cmap = "gray")
    plt.show()
    reconstruct = image + last_part
    plt.xlabel("reconstruct image")
    plt.imshow(reconstruct, cmap = 'gray')
   
# %%
#original image
image = plt.imread("../Zacos-genevre/Zacos-Geneve/masks_machine/cdn_2004_0355_A.png")
plt.imshow(image)
# %%
# extract hand mask
hand = ext.Extract_part_json("D:/M2/Stage/Byzatine/Zacos-genevre/Zacos-Geneve/ann/cdn_2004_0355_A.jpg.json",'classTitle','Part_hand')
hand_image = hand['Part_hand0'] + hand['Part_hand1']
plt.imshow(hand_image,cmap="gray")
#%%
init = np.array([0.5*np.pi, 0])
diff = symmetry(hand_image,init)

# %%
image = plt.imread("../Zacos-genevre/Zacos-Geneve/masks_human/cdn_2004_0322_A.png")
plt.imshow(image)
wings = ext.Extract_part_json("D:/M2/Stage/Byzatine/Zacos-genevre/Zacos-Geneve/ann/cdn_2004_0322_A.jpg.json",'classTitle','Part_Wings')
wings_image = wings['Part_Wings0'] + wings['Part_Wings1']
plt.imshow(wings_image)

# %%
init = np.array([0.5*np.pi, 0])
diff = symmetry(wings_image,init)

# %%
image = plt.imread("../Qijia/ds0/masks_human/Tatish 2363 patrice strat?ge  copie 2 A.png")
plt.imshow(image)
flower = ext.Extract_part_json("../Qijia/ds0/ann/Tatish 2363 patrice strat?ge  copie 2 A.jpg.json",'classTitle','Fleuron')
flower_image = flower["Fleuron0"]
plt.imshow(flower_image,cmap="gray")
# %%
init = np.array([0.5*np.pi, 30])
diff = symmetry(flower_image,init)
comput_missing_part(flower_image,diff,1024)
# %%
cross = ext.Extract_part_json("../Qijia/ds0/ann/Tatish 2451 Michel  copie A.jpg.json",'classTitle',"Croix_recroisetee")
cross_image = cross["Croix_recroisetee0"]
w,h = cross_image.shape
cross_image = transform.resize(cross_image,(w,w))
plt.imshow(cross_image,cmap="gray")

# %%
init = np.array([0.5*np.pi, -60])
diff = symmetry(cross_image,init)
comput_missing_part(cross_image,diff,2048)

# %%
