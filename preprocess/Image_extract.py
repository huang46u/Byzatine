
import json
from aiohttp import JsonPayload
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import os
import sys
sys.path.append('../')
from skimage.draw import polygon
import re

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
    with open(json_path,"r") as fcc_file:
        dictionary = json.load(fcc_file)
        w = dictionary['size']['width']
        h = dictionary['size']['height']
        for i in range(len(dictionary['objects'])):
            label = dictionary['objects'][i]['classTitle']
            if(not filter or (filter and (label in L_dict))):
                if(category):
                    cate = label.split("_")
                    if(cate[0] not in Image_dict.keys()):
                        Image_dict[cate[0]] = []
                vertices = np.array(dictionary['objects'][i]['points']['exterior'])
                vertices[:,[0,1]] = vertices[:,[1,0]]
                new_image = np.zeros((w,h))
                rr, cc = polygon(vertices[:,0], vertices[:,1], new_image.shape)
                new_image[rr,cc] = 1
                if(not (label) in label_set.keys()):
                    label_set[label] = 0
                    if(category):
                        Image_dict[cate[0]].append((label, new_image))
                    else:
                        Image_dict[label+"_" + str(label_set[label])]= new_image
                else:
                    label_set[label]+=1
                    if(category):
                        Image_dict[cate[0]].append((label+"_"+str(label_set[label]), new_image))
                    else:
                        Image_dict[label+"_" + str(label_set[label])]= new_image
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
