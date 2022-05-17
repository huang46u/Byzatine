#%%
import numpy as np
import matplotlib.pyplot as plt
from skimage.measure import label
import tools as tl
import preprocess.Image_extract as ext
import relation.histogram as his
import imp
import plot
import relation.morpho as mo
imp.reload(mo)
imp.reload(plot)
imp.reload(his)
imp.reload(tl)
#%%
json_path = "../Qijia/ds0/ann/Tatish 1754 Andr¨¦ spatharocandidat . jpg copie A.jpg.json"
Image_dict = ext.Extract_image_mask(json_path)
print(Image_dict.keys())
#%%
# Extract differnt part of the image
left_flower = Image_dict["Fleuron_1"]
cross = Image_dict["Croix_recroisetee_1"]
cross =  tl.down_sample(cross)
left_flower = tl.down_sample(left_flower)
# %%
morpho_ref_list, morpho_res_list, nece_deg, poss_deg, mean_deg = mo.morpho_relation(cross, left_flower)
plot.plot_morpho(morpho_ref_list,morpho_res_list,nece_deg,
                 poss_deg, mean_deg)
# %%
a = np.zeros((100,100))
b = np.zeros((100,100))
a[40:60, 30:50] = 1
b[20:30, 40:90 ] =1
b[20:80, 80:90 ] =1
plt.imshow(a+b, cmap = "gray") 
# %%
morpho_ref_list, morpho_res_list, nece_deg, poss_deg, mean_deg = mo.morpho_relation(a, b)
# %%
plot.plot_morpho(morpho_ref_list,morpho_res_list,nece_deg,
                 poss_deg, mean_deg, seperate = True)
# %%
# %%
json_path ="../Zacos-genevre/Zacos-Geneve/ann/cdn_2004_0539_A.jpg.json"
Image_dict = ext.Extract_image_mask(json_path)
print(Image_dict.keys())
# %%
head = Image_dict["Part_head_0"]
book = Image_dict["Object_Book_0"]
head =  tl.down_sample(head)
book = tl.down_sample(book)
# %%
morpho_ref_list, morpho_res_list, nece_deg, poss_deg, mean_deg = mo.morpho_relation(head, book)
plot.plot_morpho(morpho_ref_list,morpho_res_list,nece_deg,
                 poss_deg, mean_deg)
# %%
plt.imshow(head+book,cmap = "gray")

# %%
