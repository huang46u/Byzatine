# %%
from ast import operator
from cProfile import label
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from pyparsing import srange
from torch import greater
import tools as tl
from preprocess import *
import relation_analyse as rel
import preprocess.Image_extract as ext
import imp
import os
from os import path
import json
imp.reload(ext)
imp.reload(tl)
imp.reload(rel)

relation_path = "../Zacos-genevre/Zacos-Geneve/rel/"
json_path = "../Zacos-genevre/Zacos-Geneve/ann/cdn_2004_0322_A.jpg.json"
labels_file = open("../Classes x annotation.txt", "r")
labels_dict = [label.rstrip("\n") for label in labels_file]
Image_dict = ext.Extract_image_mask(json_path,category=True,filter=True,L_dict = labels_dict)
# %%
os.makedirs(relation_path,exist_ok=True)
# %%
ann_path = "../Zacos-genevre/Zacos-Geneve/ann/"
for file in os.listdir(ann_path):
    filename = os.path.join(ann_path, file)
    print(filename)
    Image_dict = ext.Extract_image_mask(filename, 
                                        category=True, 
                                        filter=True, 
                                        L_dict = labels_dict)
    image_dict, area_val, sorted_dict = rel.compute_area(Image_dict)
    if(len(image_dict) != 0):
        json_path = os.path.join(relation_path, file)
        data = rel.relation_analyse(Image_dict, image_dict, sorted_dict)
        if(data != None and len(data["data"]) != 0):
            with open(json_path, "w") as fp:
                json.dump(data, fp)