#%%
from tkinter import Image
import math
from turtle import right
import numpy as np
import imp
import relation.histogram as his
import preprocess.Image_extract as ext
import matplotlib.pyplot as plt
import plot
import tools as tl
imp.reload(his)
imp.reload(plot)
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
hist, bin = his.compute_histogram_force(person1, person2, n_bin =720)
hist = hist/np.max(hist)
# %%
plot.plot_histogram(hist, bin)
# %%
right_image = np.zeros(person1.shape)
left_image = np.zeros(person2.shape)
n_bin = bin.size
step = 2*math.pi/n_bin
coord1 = np.argwhere(person1>0)
coord2 = np.argwhere(person2>0)
people = person1 + person2
for i,j in np.argwhere(people==0):
        vectors = coord1 - [i,j]
        ##transpose y and x 
        vectors[:, [1, 0]] = vectors[:, [0, 1]]
        length = np.linalg.norm(vectors,axis = 1)
        cosine = np.dot(vectors,[1,0])/(length+1e-5)
        # filter the cosine value
        cosine = cosine[np.where(np.logical_and(cosine>=-1,cosine<=1))]
        angle = np.arccos(cosine)*np.where(vectors[:,1]>0,1,-1)
        if(len(cosine)>0):
            beta_min = np.min(np.arccos(cosine))
            nb_bin = int((beta_min+math.pi)//step)
            left_image[i,j] = hist[nb_bin]
            
for i,j in np.argwhere(people==0):
        vectors = coord2 - [i,j] 
        ##transpose y and x 
        vectors[:, [1, 0]] = vectors[:, [0, 1]]
        length = np.linalg.norm(vectors,axis = 1)
        cosine = np.dot(vectors,[1,0])/(length+1e-5)
        # filter the cosine value
        cosine = cosine[np.where(np.logical_and(cosine>=-1,cosine<=1))]
        angle = np.arccos(cosine)*np.where(vectors[:,1]>0,1,-1)
        if(len(cosine)>0):
            beta_min = np.min(np.arccos(cosine))
            nb_bin = int((beta_min+math.pi)//step)
            right_image[i,j] = hist[nb_bin]


# %%
plt.imshow(left_image,cmap="gray")
plt.show()
plt.imshow(right_image,cmap = "gray")
plt.show()

  # %%
image=  np.minimum(left_image,right_image)+person1+person2
plt.imshow(image, cmap = 'gray')
# %%
print(np.max(left_image))
plt.imshow(person1)

# %%
his.demo_histogram_angle(person2, person1)
# %%
a = np.zeros(person1.shape)
b = np.zeros(person1.shape)
a[40:60, 30:50] = 1
b[20:30, 40:90 ] =1
b[20:70, 80:90 ] =1
plt.imshow(a+b)                                                                                                                   
# %%
bin = 180
hist1,bin1 = his.compute_histogram_angle(a, b, n_bin = bin)
hist2,bin2 = his.compute_histogram_angle(b, a, n_bin = bin)
print(np.max(hist1),np.max(hist2))
hist1 = hist1/np.max(hist1)
hist2 = hist2/np.max(hist2)
kernel = np.array([1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1])/20.0
hist1 = np.convolve(hist1, kernel, "same")
hist2 = np.convolve(hist2, kernel, "same")
plot.plot_histogram(hist1,bin1)
plot.plot_histogram(hist2,bin2)
# %%
his.demo_histogram_angle(b,a,bin=180, step = 100)
his.demo_histogram_force(a,b,bin=180, step = 100)
# %%
step = 2*math.pi/bin
struct_element1 = np.zeros((500,500))
struct_element2 = np.zeros((500,500))
for i,j in np.argwhere(struct_element1==0):
    x = i-250
    y = j-250
    vector = np.array([y,x])
    length = math.sqrt(x**2+y**2)
    # filter the cosine value
    cosine = np.dot(np.array([y,x]),[1,0])/(length+1e-5)
    if(cosine<-1 or cosine>1): continue
    angle = np.arccos(cosine)*np.where(x<0,1,-1)
    nb_bin =  int((angle)//step)
    struct_element1[i,j] = hist1[nb_bin-1]
    struct_element2[i,j] = hist2[nb_bin-1]

# %%
plot.plot_two_image(struct_element1,struct_element2, "Histogram_ab","Histogram_ba")
# %%
right_image = np.zeros(a.shape)
left_image = np.zeros(b.shape)
n_bin = bin1.size
step = 2*math.pi/n_bin
coord1 = np.argwhere(a>0)
coord2 = np.argwhere(b>0)
c = a + b
for i,j in np.argwhere(c==0):
        vectors = [i,j] - coord1 
        ##transpose y and x 
        vectors[:, [1, 0]] = vectors[:, [0, 1]]
        length = np.linalg.norm(vectors,axis = 1)
        cosine = np.dot(vectors,[1,0])/(length+1e-5)
        # filter the cosine value
        cosine = cosine[np.where(np.logical_and(cosine>=-1,cosine<=1))]
        angle = np.arccos(cosine)*np.where(vectors[:,1]<0,1,-1)
        if(len(angle)>0):
            nb_bin = (angle)//step
            nb_bin=nb_bin.astype(np.int8)
            degrad = hist2[nb_bin]
            left_image[i,j] = np.max(degrad)     
for i,j in np.argwhere(c==0):
        vectors =[i,j] - coord2 
        ##transpose y and x 
        vectors[:, [1, 0]] = vectors[:, [0, 1]]
        length = np.linalg.norm(vectors,axis = 1)
        cosine = np.dot(vectors,[1,0])/(length+1e-5)
        # filter the cosine value
        cosine = cosine[np.where(np.logical_and(cosine>=-1,cosine<=1))]
        angle = np.arccos(cosine)*np.where(vectors[:,1]<0,1,-1)
        if(len(angle)>0):
            nb_bin = (angle)//step
            nb_bin=nb_bin.astype(np.int8)
            degrad = hist1[nb_bin]
            right_image[i,j] = np.max(degrad)   


# %%
plt.xlabel("Dilation of square")
plt.imshow(left_image+a,cmap="gray")
plt.show()
plt.xlabel("Dilation of rectangle")
plt.imshow(right_image+b,cmap = "gray")
plt.show()
plt.xlabel("Intersection")
plt.imshow(np.minimum(left_image, right_image)+a+b, cmap ="gray")
# %%


# %%
np.argwhere(c==0)
# %%
