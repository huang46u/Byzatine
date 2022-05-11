#%%
import matplotlib.pyplot as plt
import numpy as np
import imp
import plot
sys.path.append("relation")
import relation.histogram as his
import preprocess.Image_extract as ext
import relation.between as bet
import tools as tl
import sys
imp.reload(bet)
imp.reload(his)
# %%
json_path = "../Zacos-genevre/Zacos-Geneve/ann/cdn_2004_0319_A.jpg.json"
Image_dict = ext.Extract_image_mask(json_path)
print(Image_dict.keys())
# %%
person1 = Image_dict['Person_Body_0']
person2 = Image_dict['Person_Body_1']
person1 = tl.down_sample(person1)
person2 = tl.down_sample(person2)

plt.imshow(person1)
plt.show()
plt.imshow(person2)
# %%
surr, d1, d2, hist1, hist2, bin1, bin2= bet.between(person1,person2,kernel_size=30)
st1, st2 = bet.visualise_structure_element(hist1,hist2)
# %%
plot.plot_histogram(hist1,bin1)
plot.plot_histogram(hist2, bin2)
plot.plot_two_image(st1,st2,"st1","str2")
# %%
plt.xlabel("Dilation of child")
plt.imshow(d1,cmap="gray")
plt.show()
plt.xlabel("Dilation of adult")
plt.imshow(d2,cmap = "gray")
plt.show()
plt.xlabel("Intersection")
plt.imshow(surr, cmap ="gray")
# %%
a = np.zeros(person1.shape)
b = np.zeros(person1.shape)
a[40:60, 30:50] = 1
b[20:30, 40:90 ] =1
b[20:70, 80:90 ] =1
plt.imshow(a+b)       
# %%
his.demo_histogram_angle(a,b)                                                                                                         
# %%
surr, d1, d2, hist1, hist2, bin1, bin2 =bet.between(a,b,kernel_size=20)
# %%
struct_element1,struct_element2 = bet.visualise_structure_element(hist1,hist2)
plot.plot_two_image(struct_element1,struct_element2, "Histogram_ab","Histogram_ba")
#%%
plot.plot_histogram(hist1, bin1)
plot.plot_histogram(hist2, bin2)
#%%
plt.xlabel("Dilation of square")
plt.imshow(d1,cmap="gray")
plt.show()
plt.xlabel("Dilation of rectangle")
plt.imshow(d2,cmap = "gray")
plt.show()
plt.xlabel("Intersection")
plt.imshow(surr, cmap ="gray")

# %%
