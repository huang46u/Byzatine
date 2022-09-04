#%%
import matplotlib.pyplot as plt
import numpy as np
import imp
import plot
sys.path.append("relation")
import relation.histogram as his
import file_processing.Image_extract as ext
import relation.between as bet
import tools as tl
import sys
imp.reload(bet)
imp.reload(ext)
imp.reload(plot)
imp.reload(tl)
# %%
json_path = "../Zacos-genevre/Zacos-Geneve/ann/cdn_2004_0439_A.jpg.json"
Image_dict = ext.Extract_image_mask(json_path)
print(Image_dict.keys())
# %%
person1 = Image_dict['Person_Emperor_0']
person2 = Image_dict['Person_Virgin_Mary_0']
globe = Image_dict['Object_Globe_0']
person1 = tl.down_sample(person1)
person2 = tl.down_sample(person2)
globe = tl.down_sample(globe)

plt.imshow(person1)
plt.show()
plt.imshow(person2)
plt.show()
plt.imshow(globe)
# %%
# %%
surr, d1, d2, hist1, hist2, bin1, bin2 = bet.between(person1,person2,kernel_size=30)
# %%
st1, st2 = bet.visualise_structuring_element(hist1,hist2)
# %%
plot.plot_histogram(hist1,bin1, label = "Virgin_Mary_Respect_Emperor")
plot.plot_histogram(hist2, bin2, label = "Emperor_Respect_To_Virgin_Mary")
plot.plot_two_image(st1,st2,"Virgin_Mary_Respect_Emperor","Emperor_Respect_To_Virgin_Mary")
# %%
plt.xlabel("Dilation of Emperor")
plt.imshow(d1,cmap="gray")
plt.show()
plt.xlabel("Dilation of Virgin Mary")
plt.imshow(d2,cmap = "gray")
plt.show()
plt.xlabel("Intersection")
plt.imshow(surr, cmap ="gray")
# %%
object, convex_inter, ness, poss, mean = bet.eval_between(globe, surr, d1, d2)
plt.xlabel("Convex hull")
plt.imshow(convex_inter,cmap = "gray")
plt.show()
plt.xlabel("Relation between")
plt.imshow(object + person1 + person2,cmap = "gray")
print(ness, poss, mean)
# %%
a = np.zeros((100,100))
b = np.zeros((100,100))
a[40:60, 30:50] = 1
b[20:30, 40:90 ] =1
b[20:70, 80:90 ] =1
plt.imshow(a+b)       
# %%
his.demo_histogram_angle(a,b)   
his.demo_histogram_angle(b,a)                                                                                                      

# %%
surr, d1, d2, hist1, hist2, bin1, bin2 =bet.between(a,b,kernel_size=20, bin = 250)
# %%
struct_element1,struct_element2 = bet.visualise_structuring_element(hist1,hist2)
plot.plot_two_image(struct_element1,struct_element2)

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
