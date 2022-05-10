from cv2 import CAP_OPENNI_IMAGE_GENERATOR_OUTPUT_MODE, line
import numpy as np
import matplotlib.pyplot as plt
from skimage.measure import block_reduce
from skimage.morphology import binary_closing,label,disk, binary_dilation
from scipy.optimize import fmin
from sympy import comp

def plot_morpho(morpho_ref, morpho_res, nece_deg, poss_deg, mean_deg):
    for i in range(4):
        plt.figure(figsize = (10,10))
        plt.subplot(1,2,1)
        plt.imshow(morpho_res[i],cmap='gray')
        plt.subplot(1,2,2)
        plt.imshow(morpho_ref[i],cmap='gray')
        plt.show()
        print("necessity degree: "+"{:.2}\n".format(nece_deg[i])
            + "possibility degree :"+"{:.2}\n".format(poss_deg[i])
            + "means: "+"{:.2}\n".format(mean_deg[i]))
    
def plot_two_image(image1, image2, label1=None, label2=None):
    plt.figure(figsize=(10,10))
    plt.subplot(1,2,1)
    if(label1!=None):
        plt.xlabel(label1)
    plt.imshow(image1, cmap = "gray") 
    plt.subplot(1,2,2)
    if(label2!=None):
        plt.xlabel(label2)
    plt.imshow(image2, cmap = "gray")
    plt.show()
    
def plot_histogram(hist, bins):
    #Plot histogram of angle
    width = np.diff(bins)
    center = (bins[:-1] + bins[1:]) / 2
    plt.bar(center, hist/np.max(hist), align='center', width=width)
    plt.show()
    
def plot_compability(step,cen_g,compability):
    #plot compability respect to four direction
    f, ax = plt.subplots(2,2,figsize= (10,10))
    ax[0, 0].plot(np.arange(0,1,1/float(step)),compability[0])
    ax[0, 0].set_title('Left')
    ax[0, 0].set_ylim(ymin=0,ymax= 1)
    ax[0, 0].set_xlabel("compability degree, center of gravity= "+"{:.2}".format(cen_g[0]))
    ax[0, 0].set_ylabel("membership values")

    ax[0, 1].plot(np.arange(0,1,1/float(step)),compability[1])
    ax[0, 1].set_title('Right')
    ax[0, 1].set_ylim(ymin=0,ymax= 1)
    ax[0, 1].set_xlabel("compability degree, center of gravity= "+"{:.2}".format(cen_g[1]))
    ax[0, 1].set_ylabel("membership values")
    
    ax[1, 0].plot(np.arange(0,1,1/float(step)),compability[2])
    ax[1, 0].set_title('Above')
    ax[1, 0].set_ylim(ymin=0,ymax= 1)
    ax[1, 0].set_xlabel("compability degree, center of gravity= "+"{:.2}".format(cen_g[2]))
    ax[1, 0].set_ylabel("membership values")
    
    ax[1, 1].plot(np.arange(0,1,1/float(step)),compability[3])
    ax[1, 1].set_title('Below')
    ax[1, 1].set_ylim(ymin=0,ymax= 1)
    ax[1, 1].set_xlabel("compability degree, center of gravity= "+"{:.2}".format(cen_g[3]))
    ax[1, 1].set_ylabel("membership values")
    plt.tight_layout()
    plt.show()
    
def plot_sym_plan(image, symm_image, line_coord):
    copy_image = image.copy()
    copy_image[line_coord[:,0], line_coord[:,1]] = 1
    plt.figure(figsize=(10,10))  
    plt.subplot(1,2,1)
    plt.xlabel("Symetry plan") 
    plt.imshow(copy_image,cmap = 'gray')
    plt.subplot(1,2,2)
    plt.xlabel("Difference")
    plt.imshow(symm_image-image,cmap = 'gray')
    plt.show()
