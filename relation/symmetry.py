import numpy as np
import cv2
import tools as tl
import plot
from scipy.optimize import fmin
from skimage.morphology import area_opening
import matplotlib.pyplot as plt
def symetry_image(image, alpha = 0.5*np.pi, d = 0):
    d_alpha = np.rad2deg(alpha)
    u = (np.cos(alpha), np.sin(alpha))
    w,h= image.shape
    image_flip = cv2.flip(image,1)
    if(d_alpha<90):
        rotation = -2*(90-d_alpha)
    else:
        rotation = 2*(d_alpha-90)
    m_rot = cv2.getRotationMatrix2D((w/2,h/2),rotation,1)
    im_rot = cv2.warpAffine(image_flip, m_rot, (w,h)) 
    m_trans = np.float32([[1,0,2*d*u[1]],[0,1,2*d*u[0]]])
    im_last = cv2.warpAffine(im_rot, m_trans,(w,h)) 
    return im_last
    
def symetry_measure(param, image):
    alpha, d = param
    im_smy = symetry_image(image,alpha,d)
    return np.linalg.norm(image-im_smy)/(2*np.linalg.norm(image))-1

def downhill_simplex(image,init):
    return fmin(symetry_measure, x0 = init, args=(image,),initial_simplex=None)
    
def symmetry_plot_data(image, param):
    alpha, d = param
    u = np.array([np.cos(alpha), np.sin(alpha)])
    w,h = image.shape
    origin = (d*u).astype(np.int32)+ np.array([h/2,w/2]).astype(np.int32)    
    line_coord1 = tl.bresenham_ray(image,origin,alpha)
    line_coord2 = tl.bresenham_ray(image,origin,alpha+np.pi)
    symm_image = symetry_image(image,alpha ,d)
    return symm_image, line_coord1,line_coord2

def symmetry(image, param_init):
    print("init value: ",symetry_measure(param_init,image))
    symm_original, line_original1, line_original2 = symmetry_plot_data(image,param_init)
    plot.plot_sym_plan(image, symm_original,line_original1, line_original2)
    opt = downhill_simplex(image,param_init)
    print(opt)
    symm_optimized, line_optimized1, line_optimized2 = symmetry_plot_data(image,opt)
    plot.plot_sym_plan(image, symm_optimized, line_optimized1,line_optimized2)
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