
from calendar import c
import json
from aiohttp import JsonPayload
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import os
import sys

from sqlalchemy import over
sys.path.append('../')
import tools as tl
from skimage.draw import polygon
import re
def add_label(label, img, image_dict, label_set, category = False):
    label_splited = label.split('_')
    if(category):
        if(label_splited[0] not in image_dict.keys()):
            image_dict[label_splited[0]] = []
    
    if(not (label) in label_set.keys()):
        label_set[label] = 0
        if(category):
            image_dict[label_splited[0]].append((label, img))
        else:
            image_dict[label+"_" + str(label_set[label])]= img
    else:
        label_set[label]+=1
        if(category):
            image_dict[label_splited[0]].append((label+"_"+str(label_set[label]), img))
        else:
            image_dict[label+"_" + str(label_set[label])]= img
    return image_dict

def Extract_image_mask(json_path, category = False, filter = False, L_dict = None):
    """
    Parameters
    ----------
        json_path: string
            The path of the annotation file
        category: bool
            If save mask image in category
        filter: bool
            Extract all the labels or not. If it is ture, L_dict will
            be needed 
        L_dict : string array
            The array of interested label
    Returns
    -------
        Img_dict : dict
            Key: the lable of current mask
            Value: corrsponding image mask
    """

    Image_dict ={}
    label_set={}
    #Treat Person_body differently
    body_dict = {}
    with open(json_path,"r") as fcc_file:
        dictionary = json.load(fcc_file)
        w = dictionary['size']['width']
        h = dictionary['size']['height']
        for i in range(len(dictionary['objects'])):
            label = dictionary['objects'][i]['classTitle']
            if(not filter or (filter and (label in L_dict))):
                vertices = np.array(dictionary['objects'][i]['points']['exterior'])
                vertices[:,[0,1]] = vertices[:,[1,0]]
                new_image = np.zeros((w,h))
                rr, cc = polygon(vertices[:,0], vertices[:,1], new_image.shape)
                new_image[rr,cc] = 1
                #In case that one person are labeled two times with label "Person_xxx" and "Person_body",*
                #we test if one is included in another
                label_splited = label.split("_")
                if(label_splited[0] == "Person" and label_splited[1] == "Body"):
                    body_dict[label + str(len(body_dict))] = new_image
                else: 
                    Image_dict = add_label(label,new_image, Image_dict, label_set, category=True)
    if("Person" not in Image_dict.keys()):
        for l, img in body_dict.items():
             Image_dict = add_label(l, img, Image_dict,label_set, category = True )
    else:
        for l1, img1 in body_dict.items():
            overlap = False
            for l2, img2 in Image_dict['Person']:
                if(tl.inside_body(img1, img2)):
                    overlap = True
                    break
            if(not overlap):       
                Image_dict = add_label(l1, img1, Image_dict, label_set, category = True)
    return Image_dict
                    
def Extract_image_part(json_path, label, pattern):
    """
    Parameters
    ----------
        json_path: string
            The path of the annotation json file
        label : string
            The label we consider to decide whether the image will be extracted
        pattern: string
            The pattern for regular expression
    Returns
    -------
        Image_dict : dict
            Key: the required lable
            Value: corrspond image part
    """
    Image_dict ={}
    id = 0
    with open(json_path,"r") as fcc_file:
        dictionary = json.load(fcc_file)
        w = dictionary['size']['width']
        h = dictionary['size']['height']
        for i in range(len(dictionary['objects'])):
            if(re.search(pattern, dictionary['objects'][i][label])):
                vertices = np.array(dictionary['objects'][i]['points']['exterior'])
                vertices[:,[0,1]] = vertices[:,[1,0]]
                new_image = np.zeros((w,h))
                rr, cc = polygon(vertices[:,0], vertices[:,1], new_image.shape)
                new_image[rr,cc] = 1
                Image_dict[dictionary['objects'][i][label]+str(id)] = new_image
                id+=1
    return Image_dict
