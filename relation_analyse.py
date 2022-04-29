from ast import operator
from cProfile import label
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from pyparsing import srange
from torch import greater
import tools as tl
from preprocess import *
import preprocess.Image_extract as ext
import imp
import os
import json
imp.reload(ext)
imp.reload(tl)
THRESHOLD = 0.05
SYMETRY_THRESHOLD = 0.001

def compute_area(Image_dict):
    """
    Parameters: 
    ----------
        Image_dict: dictionnary
            keys: category
            value: array of tuple where every element correspond the label of 
            mask and the mask image itself
            ex: {"Person":
                    [
                        ("Person_Body_0", #image),
                        ("Person_Vrigin_Marie_0", #image),
                        ...
                    ],
                "Part":
                    [
                        ("Part_hand_0", #hand_image),
                        ("Part_hand_1", #hand_image),
                        ...
                    ],
                ...
    Returs:
    -------
        label_dict : dictionnary
            key: mask label
            value: corresponding image
        sorted_area_value: dictionnary
            key: mask label
            value: the percentage of area that the mask covers 
        sorted_image_dict:
            key: mask label in descending order of area cover
            value: corresponding percentage that the mask covers
    """
    area_dict = {}
    image_dict = {}
    for k, v in Image_dict.items():
        image_dict.update(dict(v))
    for k, v in image_dict.items():
        w, h = v.shape
        area_dict[k] = np.count_nonzero(v)/(w*h)
    sorted_area_value = sorted(area_dict.values(), reverse=True)
    sorted_image_dict = {}
    for i in sorted_area_value:
        for k in area_dict.keys():
            if(area_dict[k] == i):
                sorted_image_dict[k] = i
                break
    return image_dict, sorted_area_value, sorted_image_dict


def relation_analyse(image_dict, label_dict, area_dict):
    """
    Parameters: 
    ----------
        Image_dict: dictionnary
            keys: string, category
            value: array of tuple, every element correspond the label of 
            mask and the mask image itself
            ex: {"Person":
                    [
                        ("Person_Body_0", #image),
                        ("Person_Vrigin_Marie_0", #image),
                        ...
                    ],
                "Part":
                    [
                        ("Part_hand_0", #hand_image),
                        ("Part_hand_1", #hand_image),
                        ...
                    ],
                ...
        label_dict: dictionnary
            keys: string, mask label
            value: ndarray, corresponding image
            ex: {
                    "Person_Body_0": #image,
                    "Part_hand_0": #hand_image,
                    "Person_Virgin_Marie_"0: #Virgin_marie_image
                    ...
                }
        area_dict: dictionnary
            keys: string mask label
            value: percentage of area that the corresponding image covers
    Returs:
    -------
        data : array of dictionnary
            The result of spatial relation anlayse of image
            
    """
    
    if(len(area_dict) == 0):
        return
    data = {}
    data["data"] = []
    if(len(image_dict.keys()) < 2):
        return
    label_ingored = set(["head","Beard", "Crown","Curly"])
    
    if(len(image_dict["Person"]) == 1):
        person = image_dict["Person"][0]
        data = relation_person_center(person, data)
        head = ("Part_head", label_dict["Part_head"])
        if("Part_head" in label_dict.keys()):
            head = (person[0],label_dict["Part_head"])
        for label in ["Object", "Part"]:
            if(label in image_dict.keys()):
                object_dict = image_dict[label]
                for object in object_dict:
                    label_word = object[0].split("_")
                    if(label_word[1] not in label_ingored):
                        if(area_dict[object[0]] < area_dict[person[0]]):
                            if(tl.intersect_with_body(object[1], person[1])):
                                data = relation_person_object(head, object ,data)
                            else:
                                data = relation_person_object(person, object, data)
    
    elif(len(image_dict["Person"]) == 2):
        person1 = image_dict["Person"][0]
        person2 = image_dict["Person"][1]
        area_value1 = area_dict[person1[0]]
        area_value2 = area_dict[person2[0]]
        if(abs(area_value1 - area_value2) > THRESHOLD):
            
            if(area_value1 > area_value2):
                data = relation_person_center(person1, data)
                if(tl.inside_body(person2[1], person1[1])): 
                    data = relation_in_front_of(person1, person2, data)
                else:
                    data = relation_person_object(person1,person2,data)
            else:
                data = relation_person_center(person2, data)
                if(tl.inside_body(person1[1], person2[1])): 
                    data = relation_in_front_of(person2, person1, data)
                else:
                    data = relation_person_object(person2,person1,data)
        else:
            data = relation_person_person(person1,person2,data)
            
            
    return data

def relation_person_person(person1,person2,data):
    relation = {}
    person1_img = tl.down_sample(person1[1])
    person2_img = tl.down_sample(person2[1])
    _, _, _, _, means = tl.morpho_relation(person1_img, person2_img)
    rel = ["Rel_Left", "Rel_Right"]
    relation["obj1"] = person1[0]
    relation["obj2"] = person2[0]
    relation["rel"] = rel[np.argmax(means[:2])]
    data["data"].append(relation)
    rel = ["Rel_Left", "Rel_Right"]
    relation = {}
    relation["obj1"] = person2[0]
    relation["obj2"] = person1[0]
    relation["rel"] = rel[np.argmin(means[:2])]
    data["data"].append(relation)
    return data
    
def relation_person_center(person, data):
    relation = {}
    relation["obj1"] = person[0]
    relation["rel"] = "Rel_Center"
    data["data"].append(relation)
    return data

def relation_person_object(person, object, data):
    relation = {}
    person_img = tl.down_sample(person[1])
    object_img = tl.down_sample(object[1])
    _, _, _, _, means = tl.morpho_relation(person_img, object_img)
    rel = ["Rel_Left", "Rel_Right"]
    relation["obj1"] = person[0]
    relation["obj2"] = object[0]
    relation["rel"] = rel[np.argmax(means[:2])]
    data["data"].append(relation)
    return data

def relation_in_front_of(person1, person2, data):
    relation = {}
    relation["obj1"] = person2[0]
    relation["obj2"] = person1[0]
    relation["rel"] = "Rel_In_Front_Of"
    data["data"].append(relation)
    return data

