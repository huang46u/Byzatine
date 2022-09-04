#%%
import numpy as np
import matplotlib.pyplot as plt
from skimage.measure import label
import tools as tl
import file_processing.Image_extract as ext
import relation.histogram as his
import imp
import plot
import relation.morpho as mo
imp.reload(mo)
imp.reload(plot)
imp.reload(his)
imp.reload(tl)
#%%
json_path = "../Tatish/Tatis/ann/Tatish 1754_A.jpg.json"
Image_dict = ext.Extract_image_mask(json_path)
print(Image_dict.keys())
#%%
Image_dict
#%%
# Extract differnt part of the image
left_flower = Image_dict["Fleuron_0"]
cross = Image_dict["Cross_Patriarchal_0"]
cross =  tl.down_sample(cross)
left_flower = tl.down_sample(left_flower)
#%%
plt.imshow(left_flower+cross, cmap ="gray")
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
json_path ="../Zacos-genevre/Zacos-Geneve/ann/cdn_2004_0355_A.jpg.json"
Image_dict = ext.Extract_image_mask(json_path)
print(Image_dict.keys())
# %%
head = Image_dict["Person_Virgin_Mary_0"]
book = Image_dict["Person_Christ_Child_0"]
head =  tl.down_sample(head)
book = tl.down_sample(book)
# %%
plt.imshow(head+book, cmap= "gray")
# %%
morpho_ref_list, morpho_res_list, nece_deg, poss_deg, mean_deg = mo.morpho_relation(head, book)
plot.plot_morpho(morpho_ref_list,morpho_res_list,nece_deg,
                 poss_deg, mean_deg)
# %%
image = head+book
image = np.where(image>1, 1,image)
plt.imshow(image,cmap = "gray")

# %%
plt.imshow(book)

# %%
