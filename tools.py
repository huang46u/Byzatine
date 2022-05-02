from cv2 import CAP_OPENNI_IMAGE_GENERATOR_OUTPUT_MODE
import numpy as np
import matplotlib.pyplot as plt
import math
from skimage.measure import block_reduce
import json
import os
import cv2
from sympy import comp

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
    object = block_reduce(object,block_size=(scale,scale))
    object = (object!=0)*object
    return object
    
def bresenham_ray(image, point, theta):
    opposite = False
    steep = False
    if(theta>=math.pi): 
        theta -= math.pi
        opposite = True
    y,x = point
    w, h = image.shape
    height = 0
    coord = []
    if(theta<math.pi*0.5):
        if(theta>0.25*math.pi):
            steep = True
            theta = 0.5*math.pi - theta
        step = abs(math.tan(theta))
        if(not steep):
            while(x<h and y<w and x>=0 and y>=0):
                coord.append([y,x])
                height += step
                if(height>=0.5):
                    if(not opposite): 
                        y -= 1
                    else:
                        y += 1
                    height -= 1
                if(not opposite): 
                    x += 1
                else:
                    x -= 1
        else:
            while(x<h and y<w and x>=0 and y>=0):
                coord.append([y,x])
                height += step
                if(height>=0.5):
                    if(not opposite): 
                        x += 1
                    else:
                        x -= 1
                    height -= 1
                if(not opposite): 
                    y -= 1
                else:
                    y += 1
    elif (theta == math.pi*0.5):
        if(not opposite):
            for i in range(0,y):
                coord.append((i,x))
        else:
            for i in range(y, w):
                coord.append((i,x))
    else:
        theta = theta-math.pi
        if(theta<-0.25*math.pi):
            theta = -(theta+0.5*math.pi)
            steep = True
        step = abs(math.tan(theta))
        if(not steep):
            while(x<h and y<w and x>=0 and y>=0):
                height += step
                coord.append([y,x])
                if(height>=0.5):
                    if(not opposite): 
                        y -= 1
                    else:
                        y += 1
                    height -= 1
                if(not opposite): 
                    x -= 1
                else:
                    x += 1
        else:
            while(x<h and y<w and x>=0 and y>=0):
                height += step
                coord.append([y,x])
                if(height>=0.5):
                    if(not opposite): 
                        x += 1
                    else:
                        x -= 1
                    height -= 1
                if(not opposite): 
                    y += 1
                else:
                    y -= 1
    return np.array(coord)

def bresenham_line(image, point, theta):
    steep = False
    reverse = False
    if(theta>=math.pi): 
        theta -= math.pi
        reverse = True
    y,x = point
    w, h = image.shape
    height = 0
    coord = []
    if(theta<math.pi*0.5):
        if(theta>0.25*math.pi):
            steep = True
            theta = 0.5*math.pi - theta
        step = abs(math.tan(theta))
        if(not steep):
            while(x<h and y<w and x>=0 and y>=0):
                coord.append([y,x])
                height += step
                if(height>=0.5):
                    y -= 1   
                    height -= 1        
                x += 1
            y,x = point
            while(x<h and y<w and x>=0 and y>=0):
                coord.append([y,x])
                height += step
                if(height>=0.5):
                    y += 1
                    height -= 1
                x -= 1
        else:
            while(x<h and y<w and x>=0 and y>=0):
                coord.append([y,x])
                height += step
                if(height>=0.5):
                    x += 1
                    height -= 1
                y -= 1
            y,x = point
            while(x<h and y<w and x>=0 and y>=0):
                coord.append([y,x])
                height += step
                if(height>=0.5):
                    x -= 1
                    height -= 1
                y += 1
    elif (theta == math.pi*0.5):
            for i in range(0, w):
                coord.append((i,x))
            
    else:
        theta = theta-math.pi
        if(theta<-0.25*math.pi):
            theta = -(theta+0.5*math.pi)
            steep = True
        step = abs(math.tan(theta))
        if(not steep):
            while(x<h and y<w and x>=0 and y>=0):
                height += step
                coord.append([y,x])
                if(height>=0.5):
                    y -= 1
                    height -= 1
                x -= 1
            y,x = point
            while(x<h and y<w and x>=0 and y>=0):
                height += step
                coord.append([y,x])
                if(height>=0.5):
                    y += 1
                    height -= 1
                x += 1
        else:
            while(x<h and y<w and x>=0 and y>=0):
                height += step
                coord.append([y,x])
                if(height>=0.5):
                    x += 1
                    height -= 1
                y += 1
            y,x = point
            while(x<h and y<w and x>=0 and y>=0):
                height += step
                coord.append([y,x])
                if(height>=0.5):
                    x -= 1
                    height -= 1
                y -= 1
    coord = np.array(coord)
    if(reverse): coord = np.flip(coord,0)    
    return coord


def debug_surround(image,region,bin = 120):
    coords = np.argwhere(region)
    j=0
    for point in (coords):
        a = np.zeros(image.shape)
        intersect = []
        for i in np.arange(0,2,2/bin):
            coord = bresenham_ray(image,point,i*math.pi)
            if(image[coord[:,0],coord[:,1]].sum()==0):
                intersect.append(0)
            else: 
                a[coord[:,0],coord[:,1]]=np.max(image)
                intersect.append(1)
        intersect = np.array(intersect)
        if(j%100 ==0):
            plt.imshow(a+image)
            plt.show()
            print(intersect.sum()*(2/bin))
        j+=1
        
def debug_bresenham(image,region,bin = 120):
    coords = np.argwhere(region)
    j=0
    for point in (coords):
        a = np.zeros(image.shape)
        intersect = []
        for i in np.arange(0,1,1/bin):
            coord = bresenham_line(image,point,i*math.pi)
            if(image[coord[:,0],coord[:,1]].sum()==0):
                intersect.append(0)
            else: 
                a[coord[:,0],coord[:,1]]=np.max(image)
                intersect.append(1)
        intersect = np.array(intersect)
        if(j%100 ==0):
            plt.imshow(a+image)
            plt.show()
            print(intersect.sum()*(2/bin))
        j+=1
        
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
    return body[coord[:,0], coord[:,1]].sum()==len(coord)

def intersect_with_body(object, body):
    coord = np.argwhere(object>0)
    return body[coord[:,0], coord[:,1]].sum()>0
    
    
    
    
    
    
    