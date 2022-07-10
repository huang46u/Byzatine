from ast import operator
from cProfile import label
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from pyparsing import srange
from sympy import continued_fraction
from torch import greater
import tools as tl

import file_processing.Image_extract as ext
import relation.morpho as morpho
import relation.surround as su
import imp
import os
import json
imp.reload(ext)
imp.reload(tl)
imp.reload(su)
THRESHOLD = 0.03
SYMETRY_THRESHOLD = 0.001
label_ingored = set(["head","Beard","Curly","Veil"])
    
def relation_person_person(person1, person2, data):
    relation = {}
    person1_img = tl.down_sample(person1[1])
    person2_img = tl.down_sample(person2[1])
    _, _, _, _, means = morpho.morpho_relation(person1_img, person2_img)
    rel = ["Rel_Left", "Rel_Right"]
    relation["obj1"] = person1[0]
    relation["rel"] = rel[np.argmax(means[:2])]
    data["data"].append(relation)
    relation = {}
    relation["obj1"] = person2[0]
    relation["rel"] = rel[np.argmin(means[:2])]
    data["data"].append(relation)
    return data
    
def relation_center(object, data):
    relation = {}
    relation["obj1"] = object[0]
    relation["rel"] = "Rel_Center"
    data["data"].append(relation)
    return data

def relation_top(person, object, data):
    relation = {}
    relation["obj1"] = person[0]
    relation["obj2"] = object[0]
    relation["rel"] = "Rel_Top"
    data["data"].append(relation)
    return data

def relation_center_object_lr(person, object, data):
    relation = {}
    person_img = tl.down_sample(person[1])
    object_img = tl.down_sample(object[1])
    _, _, _, _, means = morpho.morpho_relation(person_img, object_img)
    rel = ["Rel_Left", "Rel_Right"]
    relation["obj1"] = person[0]
    relation["obj2"] = object[0]
    relation["rel"] = rel[np.argmax(means[:2])]
    data["data"].append(relation)
    return data

def relation_center_object_td(person, object, data):
    relation = {}
    person_img = tl.down_sample(person[1])
    object_img = tl.down_sample(object[1])
    _, _, _, _, means = morpho.morpho_relation(person_img, object_img)
    rel = ["Rel_Top", "Rel_Down"]
    relation["obj1"] = person[0]
    relation["obj2"] = object[0]
    relation["rel"] = rel[np.argmax(means[-2::])]
    data["data"].append(relation)
    return data

def relation_center_object_4(cross, object, data):
    relation = {}
    cross_img = tl.down_sample(cross[1])
    object_img = tl.down_sample(object[1])
    _, _, _, _, means = morpho.morpho_relation(cross_img, object_img)
    rel = ["Rel_Left", "Rel_Right", "Rel_Above", "Rel_Below"]
    relation["obj1"] = cross[0]
    relation["obj2"] = object[0]
    relation["rel"] = rel[np.argmax(means)]
    data["data"].append(relation)
    return data

def relation_in_front_of(person1, person2, data):
    relation = {}
    relation["obj1"] = person1[0]
    relation["obj2"] = person2[0]
    relation["rel"] = "Rel_In_Front_Of"
    data["data"].append(relation)
    return data

def analyse_one_person_relation(person, image_dict, label_dict, area_dict, data):
    head = ("Part_head", label_dict["Part_head"])
    if("Part_head" in label_dict.keys()):
        head = (person[0],label_dict["Part_head"])
    for label in ["Object", "Part", "Nimbus"]:
        if(label in image_dict.keys()):
            object_dict = image_dict[label]
            for object in object_dict:
                label_word = object[0].split("_")
                if(label_word[1] not in label_ingored):
                    #determine if "hand" is belong to current person
                    if(label_word[1] == "hand" and not 
                       tl.intersect_with_body(object[1],person[1])):
                        continue
                    #Test the surround relation between nimbus, crown and head
                    if(label_word[1] == "Crown" or label_word[0]=="Nimbus"):
                        nece, poss, mean = su.eval_surround(object[1], head[1])
                        if(mean>0.1):
                            data = relation_top(head, object, data)
                        continue
                    """if(area_dict[object[0]] < area_dict[person[0]]):
                        if(tl.intersect_with_body(object[1], person[1])):
                            data = relation_center_object_lr(head, object ,data)
                        else:
                            data = relation_center_object_4(person, object, data)"""
                    if(tl.intersect_with_body(object[1], person[1])):
                            data = relation_center_object_lr(head, object ,data)
                    else:
                        data = relation_center_object_4(person, object, data)        
    return data

def analyse_croix_object_relation(croix, image_dict, data):
    data = relation_center(croix, data)
    for label in ["Fleuron", "Step", "Ornament"]:
        if(label in image_dict.keys()):
            object_dict = image_dict[label]
            for object in object_dict:
                data = relation_center_object_4(croix, object, data)
    return data

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
        sorted_array_dict: dictionnary
            keys: string, category
                value: array of tuple, every element contain the label of 
                mask and the mask image itself. The order is decreasing according
                to the area covered by the mask
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
        label_dict : dictionnary
            keys: string, mask label
            value: ndarray, corresponding image
            ex: {
                    "Person_Body": #image,
                    "Part_hand": #hand_image,
                    "Person_Virgin_Marie": #Virgin_marie_image
                    ...
                }
        sorted_image_dict:
            key: mask label in descending order according to the 
            area covered by the mask
            value: the percentage of the area coverage
    """
    area_dict = {}
    label_dict = {}
    sorted_array_dict = {}
    for k, v in Image_dict.items():
        label_dict.update(dict(v))
    for k, v in label_dict.items():
        w, h = v.shape
        area_dict[k] = np.count_nonzero(v)/(w*h)
    sorted_area_value = sorted(area_dict.values(), reverse=True)
    sorted_image_dict = {}
    for i in sorted_area_value:
        for k in area_dict.keys():
            if(area_dict[k] == i):
                sorted_image_dict[k] = i
                break
    for label in sorted_image_dict:
        l = label.split("_")
        if(l[0] not in sorted_array_dict.keys()):
            sorted_array_dict[l[0]] = []
        sorted_array_dict[l[0]].append((label, label_dict[label]))
            
    return sorted_array_dict, label_dict, sorted_image_dict

def relation_analyse(image_dict, label_dict, area_dict):
    """
    Parameters: 
    ----------
        image_dict: dictionnary
            keys: string, category
            value: array of tuple, every element contain the label of 
            mask and the mask image itself. The order is decreasing according
            to the area covered by the mask
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
                    "Person_Body": #image,
                    "Part_hand": #hand_image,
                    "Person_Virgin_Marie": #Virgin_marie_image
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
    data = {}
    data["data"] = []
    if(len(image_dict.keys()) < 2):
        return
    if("Person" in image_dict.keys()):
        if(len(image_dict["Person"]) == 1):
            person = image_dict["Person"][0]
            data = relation_center(person, data)
            data = analyse_one_person_relation(person, image_dict,label_dict, area_dict, data)
        
        elif(len(image_dict["Person"]) == 2):
            #Person1's area is bigger than Person2
            person1 = image_dict["Person"][0]
            person2 = image_dict["Person"][1]
            person_area1 = area_dict[person1[0]]
            person_area2 = area_dict[person2[0]]
            #relation between two people
            #If one person's head is totally included by another person's area
            #we believe that the first person is in front of another one
            if(abs(person_area1 - person_area2) > THRESHOLD):
                    data = relation_center(person1, data)
                    if(tl.inside_body(person2[1], person1[1])): 
                        data = relation_in_front_of(person1, person2, data)
                    else:
                        data = relation_center_object_lr(person1,person2,data)
                    data = analyse_one_person_relation(person1, image_dict,label_dict,area_dict,data)
            else:
                data = relation_person_person(person1,person2,data)
    elif("Cross" in image_dict.keys()):
        croix = image_dict["Cross"][0]
        data = analyse_croix_object_relation(croix, image_dict,data)
    return data

def relation_compare(file1, file2):
    rel1, rel2 = {}
    with open(file1,"r") as rel_file1:
        rel1 = json.load(rel_file1)
    with open(file2,"r") as rel_file2:
        rel2 = json.load(rel_file2)

def relation_extract_to_text(rel_data, filename):
    with open(filename, 'w') as f:
        for rel in rel_data["data"]:
            if rel["rel"] == "Rel_Center":
                f.write(rel["obj1"] + " is in the center of image\n")
                continue
            if rel["rel"] == "Rel_Left":
                if("obj2" in rel.keys()):
                    f.write(rel["obj2"] + " is on the left of " + rel["obj1"]+"\n")
                else:
                    f.write(rel["obj1"]+ " is on the left of the image \n")
                continue
            if rel["rel"] == "Rel_Right":
                if("obj2" in rel.keys()):
                    f.write(rel["obj2"] + " is on the right of " + rel["obj1"] +"\n")
                else:
                    f.write(rel["obj1"]+ " is on the right of the image \n")
                continue
            
            if rel["rel"] == "Rel_Top":
                f.write(rel["obj2"] + " is on the top of " + rel["obj1"] +"\n")
                continue
            if rel["rel"] == "Rel_Down":
                f.write(rel["obj2"] + "is on the down of " + rel["obj1"] +"\n")
                continue
            if rel["rel"] == "Rel_In_Front_Of":
                f.write(rel["obj2"] + "is in the front of " + rel["obj1"] +"\n")
                continue
