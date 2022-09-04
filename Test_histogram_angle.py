#%%
import numpy as np
import matplotlib.pyplot as plt
from skimage.measure import label
import tools as tl
import file_processing.Image_extract as ext
import relation.histogram as his
import imp
import plot
imp.reload(plot)
imp.reload(his)
imp.reload(tl)
#%%
json_path = "../Qijia/ds0/ann/Tatish 2706 A 2016 copie.jpg.json"
Image_dict = ext.Extract_image_mask(json_path)
print(Image_dict.keys())
#%%
# Extract differnt part of the image
left_flower = Image_dict["Fleuron_0"]
cross = Image_dict["Croix_recroisetee_0"]
cross =  tl.down_sample(cross)
left_flower = tl.down_sample(left_flower)
# %%
his.demo_histogram_force(cross,left_flower,"Cross", "Flower", "Histogram of angle", bin = 360)
his.demo_histogram_force(cross,left_flower,"Cross", "Flower", "Histogram of angle", bin = 180)

# %%
a = np.zeros((100,100))
b = np.zeros((100,100))
a[40:60, 30:50] = 1
b[20:30, 40:90 ] =1
b[20:80, 80:90 ] =1
plt.imshow(a+b, cmap = "gray")       
# %%
his.demo_histogram_angle(a,b,"Object A", "Object B", "Histogram of angle, bin = 180", step = 100)        
his.demo_histogram_angle(a,b,"Object A", "Object B", "Histogram of angle, bin = 180", step = 100)    

# %%
