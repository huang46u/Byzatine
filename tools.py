from cv2 import CAP_OPENNI_IMAGE_GENERATOR_OUTPUT_MODE
import numpy as np
import matplotlib.pyplot as plt
import math
from skimage.measure import block_reduce
import json
import os
import cv2
from sympy import comp
from skimage.draw import line as skimage_line
def unique_value(img):
    return np.unique(img.reshape(-1,img.shape[1]))[1:]

def down_sample(object, scale=8):
    """
    Down sample image by scale
    Args:
        object: image to be down sample
        scale: down sample scale, default is 8
    Return:
        Down sampled image
    """
    object = block_reduce(object,block_size=(scale,scale),func = np.mean)
    object = (object!=0)*object
    return object
    
def line(theta, length):
    x1, y1 = (0, 0)
    x2 = int(round(length * math.cos(theta)))
    y2 = int(round(length * -math.sin(theta)))
    ii, jj = skimage_line(y1, x1, y2, x2)
    return ii, jj

def bresenham_ray(image, point, theta):
    theta = math.fmod(theta, 2 * np.pi)
    w, h = image.shape
    rho = 2 * max(w, h)
    ii, jj = line(theta, rho)
    #translate the origin to point
    ii += point[0]
    jj += point[1]
    ii, jj = restrict(ii,jj, h, w)
    return ii, jj
    
def restrict(ii, jj, height, width):
    mask = (ii >= 0) & (ii < width) & (jj >= 0) & (jj < height)
    return ii[mask], jj[mask]

def parse_json(filename):
    """Parse json file 
    Args:
        filename: the json file name;
    Return:
        dictionnary
    """
    with open(filename, 'r') as fcc_file:
        dct = json.load(fcc_file)
        inv_dict = {int(v[0]): k for k, v in dct.items()}
        dct = {k : int(v[0]) for k,v in dct.items()}
        return dct,inv_dict
 
def read_images(dirPath):
    images = []
    for file in os.listdir(dirPath):
        if(os.path.isfile(os.path.join(dirPath, file))==True):
            base = os.path.basename(file)
            name = dirPath+'\\' +base
            img = cv2.imread(name)
            images.append(img)
    return np.array(images)  
        
def inside_body(object, body):
    """
    Test if objects are inside the body
    
    Args:
        object : object to test if it is insed the body
        body: image of person
    Return:
        insede: bool
        if the object is totally inside the body
    """
    coord = np.argwhere(object>0)
    a = body[coord[:,0], coord[:,1]].sum()
    b = len(coord)
    c = a/b
    return c > 0.98

def intersect_with_body(object, body):
    coord = np.argwhere(object>0)
    return body[coord[:,0], coord[:,1]].sum()>0
    
    
    
    
    
    
    