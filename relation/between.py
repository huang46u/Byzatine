import numpy as np
import imp
import histogram as his
import preprocess.Image_extract as ext
import matplotlib.pyplot as plt
import plot
import math
imp.reload(his)

def fuzzy_histogram_dilation(image, hist):
    coord = np.argwhere(image>0)
    dilation_image = np.zeros(image.shape)
    step = 2*math.pi / hist.size
    for i,j in np.argwhere(image==0):
            vectors = [i,j] - coord 
            ##transpose y and x 
            vectors[:, [1, 0]] = vectors[:, [0, 1]]
            length = np.linalg.norm(vectors,axis = 1)
            cosine = np.dot(vectors,[1,0])/(length+1e-5)
            # filter the cosine value
            cosine = cosine[np.where(np.logical_and(cosine>=-1,cosine<=1))]
            angle = np.arccos(cosine)*np.where(vectors[:,1]<0,1,-1)
            if(len(angle)>0):
                nb_bin = (angle+math.pi)//step
                nb_bin = nb_bin.astype(np.uint8)
                degrad = hist[nb_bin]
                dilation_image[i,j] = np.max(degrad)  
    return dilation_image

def visualise_structure_element(hist1,hist2,size=500):
    step = 2*math.pi/hist1.size
    struct_element1 = np.zeros((500,500))
    struct_element2 = np.zeros((500,500))
    for i,j in np.argwhere(struct_element1==0):
        x = i-size/2
        y = j-size/2
        length = math.sqrt(x**2+y**2)
        # filter the cosine value
        cosine = np.dot(np.array([y,x]),[1,0])/(length+1e-5)
        if(cosine<-1 or cosine>1): continue
        angle = np.arccos(cosine)*np.where(x<0,1,-1)
        nb_bin =  int((angle+math.pi)//step)
        struct_element1[i,j] = hist1[nb_bin-1]
        struct_element2[i,j] = hist2[nb_bin-1]
    return struct_element1, struct_element2

def between(image1, image2,kernel_size = 1, bin = 180):
    hist1,bin1 = his.compute_histogram_angle(image1, image2, n_bin = bin)
    hist2,bin2 = his.compute_histogram_angle(image2, image1, n_bin = bin)
    hist1 = hist1/np.max(hist1)
    hist2 = hist2/np.max(hist2)
    kernel = np.ones(kernel_size)/kernel_size
    hist1 = np.convolve(hist1, kernel, "same")
    hist2 = np.convolve(hist2, kernel, "same")
    dilation_image1 = fuzzy_histogram_dilation(image1,hist1)
    dilation_image2 = fuzzy_histogram_dilation(image2,hist2)     
    surr = np.minimum(dilation_image1,dilation_image2)
    surr[np.where(image1)]=1
    surr[np.where(image2)]=1
    return surr, dilation_image1, dilation_image2, hist1,hist2,bin1,bin2

